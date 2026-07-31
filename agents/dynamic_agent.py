from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage
from stm.state import AgentState
from utils import get_llm
from typing import List, Callable, Any, Dict, Optional
import inspect
import json
import logging

logger = logging.getLogger(__name__)

# Tools that accept an injected webhook_url argument
_WEBHOOK_TOOL_NAMES = {
    "send_gmail_message",
    "create_calendar_event",
    "call_marketplace_connector",
    "send_discord_message",
    "trigger_external_workflow",
}

class DynamicAgent:
    def __init__(self, name: str, system_prompt: str, tools: List[Callable] = None, temperature: float = 0, guardrails: Optional[str] = None, agent_id: str = "default", webhook_url: Optional[str] = None):
        self.name = name
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.guardrails = guardrails
        self.webhook_url = webhook_url
        self.tools = tools or []
        self.llm = get_llm(temperature=temperature)
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    def _tool_accepts_arg(self, tool: Any, arg_name: str) -> bool:
        """Return True if the tool's callable signature includes arg_name."""
        try:
            func = getattr(tool, "func", None) or getattr(tool, "coroutine", None) or tool
            return arg_name in inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False

    def node_func(self, state: AgentState):
        """
        Dynamic node function that handles tool loops and shared memory context.
        """
        logger.info(f"--- Executing Agent: {self.name} ---")
        messages = list(state.get('messages', []))
        # Copy so we don't mutate shared state in-place before the reducer runs
        extracted_data = dict(state.get("extracted_data") or {})
        
        # 1. Inject Shared Memory into Context
        memory_context = ""
        if extracted_data:
            memory_context = f"\n\n[SHARED MEMORY / EXTRACTED DATA]:\n{json.dumps(extracted_data, indent=2)}\nUse this data for your analysis if relevant."
        
        # 2. Construct dynamic system prompt with Guardrails
        guardrails_context = ""
        if self.guardrails:
            guardrails_context = f"\n\n[GUARDRAILS / CONSTRAINTS]:\n{self.guardrails}"
            
        full_system_prompt = f"{self.system_prompt}\nYour Name: {self.name}{guardrails_context}{memory_context}"
        system_msg = SystemMessage(content=full_system_prompt)
        
        # 3. Execution Loop (Handles multiple tool calls if needed)
        node_new_messages = []
        current_iter = 0
        max_iters = 5
        
        while current_iter < max_iters:
            # Prepare call
            call_messages = [system_msg] + messages + node_new_messages
            response = self.llm.invoke(call_messages)
            # Tag AI messages with the agent that produced them
            if isinstance(response, AIMessage):
                response.name = self.name
                response.additional_kwargs = {
                    **(response.additional_kwargs or {}),
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                }
            
            # Add this response to our local tracking
            node_new_messages.append(response)
            
            # Check for tool calls
            if not response.tool_calls:
                break
                
            # Execute Tools
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                args = dict(tool_call.get('args') or {})
                logger.info(f"Agent {self.name} calling tool: {tool_name} with {args}")
                
                selected_tool = next((t for t in self.tools if t.name == tool_name), None)
                if selected_tool:
                    try:
                        # Special handling for knowledge base search to inject current agent's ID
                        if tool_name == "search_knowledge_base":
                            args["agent_id"] = self.agent_id
                        
                        # Only inject webhook_url into tools that accept it
                        if self.webhook_url and (
                            tool_name in _WEBHOOK_TOOL_NAMES
                            or self._tool_accepts_arg(selected_tool, "webhook_url")
                        ):
                            args["webhook_url"] = self.webhook_url
                        
                        tool_result = selected_tool.invoke(args)
                        result_str = str(tool_result)
                        node_new_messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=result_str,
                            name=tool_name,
                        ))
                    except Exception as e:
                        logger.exception(f"Tool {tool_name} failed for agent {self.name}")
                        node_new_messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=f"Error executing tool: {str(e)}",
                            name=tool_name,
                        ))
                else:
                    node_new_messages.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=f"Tool '{tool_name}' not found for this agent.",
                        name=tool_name,
                    ))
            
            current_iter += 1

        # 4. Global JSON Extraction (Scan all new messages for data to sync to Shared Memory)
        for msg in node_new_messages:
            content = msg.content
            if isinstance(content, list):
                content = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
            
            if isinstance(content, str) and "{" in content and "}" in content:
                try:
                    start = content.index("{")
                    end = content.rindex("}") + 1
                    json_data = json.loads(content[start:end])
                    if isinstance(json_data, dict):
                        data_to_merge = json_data.get("extracted_data", json_data)
                        if isinstance(data_to_merge, dict):
                            extracted_data.update(data_to_merge)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        logger.info(f"--- Finished Agent: {self.name} ---")
        return {
            "messages": node_new_messages,
            "extracted_data": extracted_data
        }
