"""Node functions for the deterministic front half of the graph:
transcribe -> (translate) -> extract entities -> classify.

The tool-calling agent half lives in graph.py, since it needs
langgraph.prebuilt's ToolNode/tools_condition alongside it.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import get_llm, get_nlp, get_whisper_model
from .state import CallState

URGENT_KEYWORDS = ["urgent", "broken", "cancel", "refund", "angry", "asap"]


def transcribe_node(state: CallState) -> dict:
    model = get_whisper_model()
    result = model.transcribe(state["audio_path"])
    return {"transcript": result["text"], "language": result["language"]}


def route_after_transcribe(state: CallState) -> Literal["translate", "extract_entities"]:
    return "translate" if state["language"] != "en" else "extract_entities"


_translate_chain = None


def get_translate_chain():
    global _translate_chain
    if _translate_chain is None:
        prompt = ChatPromptTemplate(
            [
                ("system", "Translate the given text to English. Return only the translated text, nothing else."),
                ("human", "Translate this {lang} text:\n{text}"),
            ]
        )
        _translate_chain = prompt | get_llm() | StrOutputParser()
    return _translate_chain


def translate_node(state: CallState) -> dict:
    chain = get_translate_chain()
    english = chain.invoke({"lang": state["language"], "text": state["transcript"]})
    # StrOutputParser can return a str subclass depending on langchain-core
    # version, so cast explicitly before this hits spaCy's tokenizer.
    return {"english_text": str(english)}


def extract_entities_node(state: CallState) -> dict:
    text = state.get("english_text") or state["transcript"]
    nlp = get_nlp()
    doc = nlp(text)

    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    lowered = text.lower()
    found_keywords = [kw for kw in URGENT_KEYWORDS if kw in lowered]

    return {"entities": {"persons": persons, "urgent_keywords": found_keywords}}


class Classification(BaseModel):
    classification: Literal["Open", "Closed", "Urgent"] = Field(description="Call outcome category")
    reason: str = Field(description="One-line reason for the classification")
    summary: str = Field(description="2-3 sentence summary of the call")


_classify_chain = None


def get_classify_chain():
    global _classify_chain
    if _classify_chain is None:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You triage customer support call transcripts into one of three buckets.\n"
                    "Closed = the issue was resolved on the call itself.\n"
                    "Open = needs a human follow-up, but nothing urgent.\n"
                    "Urgent = the caller is angry, or there's a safety/legal/refund risk that "
                    "needs attention right away.\n"
                    "Base the call on the transcript and the extracted entities together.",
                ),
                ("human", "Transcript:\n{transcript}\n\nEntities:\n{entities}"),
            ]
        )
        _classify_chain = prompt | get_llm().with_structured_output(Classification)
    return _classify_chain


def classify_node(state: CallState) -> dict:
    chain = get_classify_chain()
    text = state.get("english_text") or state["transcript"]
    result = chain.invoke({"transcript": text, "entities": state["entities"]})
    return {
        "classification": result.classification,
        "reason": result.reason,
        "summary": result.summary,
    }


def kickoff_agent_node(state: CallState) -> dict:
    """Turns the classification result into the first message the agent sees."""
    content = (
        f"Call classification: {state['classification']}\n"
        f"Reason: {state['reason']}\n"
        f"Summary: {state['summary']}\n"
        f"Entities: {state['entities']}\n\n"
        "Take the single most appropriate action using your tools: "
        "create_urgent_ticket for Urgent calls, schedule_followup for Open calls, "
        "or close_ticket for Closed calls. Then reply with one confirmation sentence."
    )
    return {"messages": [HumanMessage(content=content)]}
