from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """The shared state of the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    extracted_data: dict | None
