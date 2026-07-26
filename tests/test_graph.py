"""Tests for the call-classifier graph.

Nothing here hits a real API, downloads Whisper weights, or needs a spaCy
model on disk. Every heavy dependency is monkeypatched at the getter
functions defined in app/config.py, which is the whole reason those
getters are lazy in the first place.

Run with: python -m pytest
"""

from langchain_core.messages import AIMessage

from app import db, graph, nodes


class FakeWhisperModel:
    def transcribe(self, audio_path):
        return {
            "text": "mera product kharab hai aur mujhe refund chahiye abhi",
            "language": "hi",
        }


class FakeTranslateChain:
    def invoke(self, inputs):
        return "Hello, my product is broken and I need a refund right now."


class FakeEnt:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNLP:
    def __call__(self, text):
        ents = [FakeEnt("Asha", "PERSON")] if "Asha" in text else []
        return FakeDoc(ents)


class FakeClassifyResult:
    def __init__(self, classification, reason, summary):
        self.classification = classification
        self.reason = reason
        self.summary = summary


class FakeClassifyChain:
    def __init__(self, result):
        self.result = result

    def invoke(self, inputs):
        return self.result


class FakeAgentLLM:
    """First call requests a tool, second call gives a final answer.

    Mirrors how a real tool-calling model behaves across the agent/tools
    loop, so tools_condition gets exercised in both directions.
    """

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_urgent_ticket",
                        "args": {"reason": "Angry customer wants a refund for a broken product", "customer_name": "unknown"},
                        "id": "call_1",
                    }
                ],
            )
        return AIMessage(content="Urgent ticket created and escalated to a human agent.")


def _use_temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tickets.db")


def test_route_after_transcribe_non_english():
    assert nodes.route_after_transcribe({"language": "hi"}) == "translate"


def test_route_after_transcribe_english():
    assert nodes.route_after_transcribe({"language": "en"}) == "extract_entities"


def test_tools_log_to_sqlite(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    from app.tools import close_ticket, create_urgent_ticket

    msg = create_urgent_ticket.invoke({"reason": "test reason", "customer_name": "Asha"})
    assert "Asha" in msg

    close_ticket.invoke({"summary": "resolved on the call"})

    rows = db.fetch_all()
    assert [r["action"] for r in rows] == ["URGENT_TICKET", "CLOSED"]


def test_extract_entities_node_finds_person_and_keywords(monkeypatch):
    monkeypatch.setattr(nodes, "get_nlp", lambda: FakeNLP())
    state = {"english_text": "Asha called about a broken order and wants a refund asap."}

    result = nodes.extract_entities_node(state)

    assert result["entities"]["persons"] == ["Asha"]
    assert set(result["entities"]["urgent_keywords"]) == {"broken", "refund", "asap"}


def test_graph_compiles():
    # Compiling only builds the graph structure, so this needs none of
    # the heavy deps or an API key.
    app = graph.build_graph()
    assert app is not None


def test_end_to_end_pipeline_routes_urgent_call_to_ticket_tool(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)

    monkeypatch.setattr(nodes, "get_whisper_model", lambda: FakeWhisperModel())
    monkeypatch.setattr(nodes, "get_translate_chain", lambda: FakeTranslateChain())
    monkeypatch.setattr(nodes, "get_nlp", lambda: FakeNLP())

    fake_result = FakeClassifyResult(
        classification="Urgent",
        reason="Customer is angry about a broken product and wants a refund.",
        summary="Caller reported a broken product and demanded an immediate refund.",
    )
    monkeypatch.setattr(nodes, "get_classify_chain", lambda: FakeClassifyChain(fake_result))

    # Reuses the same FakeAgentLLM instance across calls. A lambda that
    # builds a fresh one each time would reset `.calls` to 0 on every
    # agent visit, so the loop would never produce a final answer and
    # would hit LangGraph's recursion limit instead.
    fake_agent_llm = FakeAgentLLM()
    monkeypatch.setattr(graph, "get_llm", lambda: fake_agent_llm)

    app = graph.build_graph()
    final_state = app.invoke({"audio_path": "sample/sample_calls/open_hindi.mp3", "messages": []})

    assert final_state["language"] == "hi"
    assert final_state["english_text"].startswith("Hello, my product is broken")
    assert final_state["entities"]["urgent_keywords"]
    assert final_state["classification"] == "Urgent"

    rows = db.fetch_all()
    assert len(rows) == 1
    assert rows[0]["action"] == "URGENT_TICKET"

    last_message = final_state["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert not last_message.tool_calls
    assert "escalated" in last_message.content.lower()
