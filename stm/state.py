from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
from langchain_core.messages import BaseMessage
import operator


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer that shallow-merges extracted_data across parallel / sequential nodes."""
    merged: Dict[str, Any] = {}
    if left:
        merged.update(left)
    if right:
        merged.update(right)
    return merged


class AgentState(TypedDict):
    """The shared state of the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    extracted_data: Annotated[Optional[Dict[str, Any]], merge_dicts]
