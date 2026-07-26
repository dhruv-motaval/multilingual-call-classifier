from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph.message import add_messages


class CallState(TypedDict, total=False):
    """Shared state passed between every node in the graph.

    Nodes only return the keys they update; LangGraph merges the rest
    (messages merges via add_messages instead of being overwritten).
    """

    audio_path: str
    transcript: str
    language: str
    english_text: str
    entities: Dict[str, Any]
    classification: str
    reason: str
    summary: str
    messages: Annotated[List[Any], add_messages]
