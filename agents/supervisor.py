from langchain_core.messages import SystemMessage, AIMessage
from utils import get_llm
from typing import List
import logging
import re

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self, name: str, system_prompt: str, workers: List[str]):
        self.name = name
        self.workers = workers
        self.system_prompt = (
            f"{system_prompt}\n\n"
            "You are the Lead Orchestrator. You manage a team of specialized workers to solve complex user requests.\n"
            f"Your workers are: {', '.join(workers)}.\n\n"
            "RULES:\n"
            "1. Analyze the conversation history carefully.\n"
            "2. If a worker is needed, provide a clear instruction for them and end your message with 'NEXT: [Worker Name]'.\n"
            "3. If the task is complete and you have the final answer for the user, end your message with 'NEXT: FINISH'.\n"
            "4. Never hallucinate worker names. Only use the ones listed above."
        )
        self.llm = get_llm(temperature=0)

    def _resolve_destination(self, content: str) -> str:
        """Parse NEXT: command and map to a known worker id or FINISH."""
        if "NEXT:" not in content:
            return "FINISH"

        cmd_part = content.split("NEXT:")[-1].strip()
        # Take the first token / bracketed name after NEXT:
        token = re.split(r"[\s,.;]+", cmd_part, maxsplit=1)[0].strip("[]()\"'")
        if not token:
            return "FINISH"

        upper = token.upper()
        if upper == "FINISH":
            return "FINISH"

        # Exact (case-insensitive) match only — avoid substring false positives
        for w in self.workers:
            if w.upper() == upper:
                return w

        # Fallback: full worker id contained as a whole word in the command line
        for w in self.workers:
            if re.search(rf"\b{re.escape(w)}\b", cmd_part, re.IGNORECASE):
                return w

        logger.warning(f"Supervisor could not resolve worker from '{cmd_part}'; defaulting to FINISH")
        return "FINISH"

    def node_func(self, state):
        messages = state.get('messages', [])
        
        response = self.llm.invoke([SystemMessage(content=self.system_prompt)] + list(messages))
        content = response.content if isinstance(response.content, str) else str(response.content)
        
        dest = self._resolve_destination(content)

        if isinstance(response, AIMessage):
            response.name = self.name
            response.additional_kwargs = {
                **(response.additional_kwargs or {}),
                "agent_name": self.name,
                "next": dest,
            }
        
        return {
            "next": dest,
            "messages": [response]
        }
