from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils import get_llm
from typing import List, Optional
import json

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

    def node_func(self, state):
        messages = state.get('messages', [])
        
        # We invoke the LLM to get reasoning + delegation
        response = self.llm.invoke([SystemMessage(content=self.system_prompt)] + messages)
        content = response.content
        
        # Extract the 'NEXT:' command
        dest = "FINISH"
        if "NEXT:" in content:
            cmd_part = content.split("NEXT:")[-1].strip().upper()
            if "FINISH" in cmd_part:
                dest = "FINISH"
            else:
                for w in self.workers:
                    if w.upper() in cmd_part:
                        dest = w
                        break
        
        return {
            "next": dest,
            "messages": [response]
        }
