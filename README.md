# LangGraph Multi-Agent Call Classifier

A rebuild of my [Multilingual Call Classifier](https://github.com/dhruv-motaval/multilingual-call-classifier)
as a proper LangGraph graph instead of a linear notebook script. Audio
still goes in and a classified, actioned call comes out, but the
difference is how it gets there. A deterministic pipeline hands off to a
real tool-calling agent that decides for itself what to do about the
call, using the same StateGraph / ToolNode / tools_condition primitives
as my [LangGraph Tool-Calling Agent](https://github.com/dhruv-motaval/LangGraph-Tool-Calling-Agent)
project.

## What changed from the original notebook

The original `main.ipynb` was a straight-line script: transcribe,
translate, extract entities, classify, print a string. It worked, but
nothing after classification actually did anything.

- The linear chain is now a LangGraph `StateGraph`, so the transcribe →
  translate → extract → classify flow is explicit and conditionally
  routed (English calls skip translation entirely).
- Classification uses structured output (`with_structured_output(Classification)`),
  so `classification` / `reason` / `summary` come back as typed fields
  instead of parsed free text.
- A tool-calling action agent runs after classification. It isn't a
  hardcoded `if classification == "Urgent"`. The agent gets the
  classification, transcript, and three tools (`create_urgent_ticket`,
  `schedule_followup`, `close_ticket`) and decides which one to call,
  looping through `agent -> tools -> agent` via `tools_condition` until
  it's done.
- Tool calls write to a small SQLite log (`tickets.db`), so there's a
  real, inspectable side effect instead of a print statement.
- Tests don't need an API key or GPU. Whisper, spaCy, and the Groq
  client are all faked out at the boundary, so the graph's routing and
  tool-calling logic are verified on every run, offline, in under a
  second.

## Architecture

```
START
  |
  v
transcribe --(non-English)--> translate --+
  |                                        |
  (English)                                v
  +-------------------------------> extract_entities
                                            |
                                            v
                                        classify
                                            |
                                            v
                                      kickoff_agent
                                            |
                                            v
                            agent <--tools_condition--> tools
                             |          (loops until no tool calls)
                             v
                            END
```

| Stage | Node(s) | What it does |
|---|---|---|
| Transcription | `transcribe` | Whisper speech-to-text + language detection |
| Routing | conditional edge | English calls skip straight to entity extraction |
| Translation | `translate` | Groq LLM translates non-English transcripts to English |
| Entity extraction | `extract_entities` | spaCy NER for person names + a small urgency-keyword scan |
| Classification | `classify` | Structured-output LLM call → Open / Closed / Urgent + reason + summary |
| Action agent | `agent` / `tools` | Tool-calling loop that decides and executes the actual next step |

## Project layout

```
app/
  config.py   # env vars + lazy-loaded Whisper/spaCy/LLM getters
  state.py    # CallState TypedDict shared across all nodes
  nodes.py    # transcribe / translate / extract_entities / classify
  tools.py    # create_urgent_ticket, schedule_followup, close_ticket
  db.py       # SQLite logging for tool calls
  graph.py    # StateGraph wiring + the agent/tools loop
run.py        # CLI: run the graph on one audio file
tests/
  test_graph.py  # routing, tools, and a full fake end-to-end run
sample/
  sample_calls/  # drop your own .mp3/.wav test calls here
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
# then fill in GROQ_API_KEY
```

`ffmpeg` needs to be installed system-side for Whisper to read audio
files (`brew install ffmpeg` / `apt install ffmpeg`).

## Running it

```bash
python run.py sample/sample_calls/test-5.mp3
```

This prints the transcript, the detected language (and translation, if
applicable), extracted entities, the classification, and the agent's
tool call and confirmation message. Every ticket the agent creates also
lands in `tickets.db`:

```bash
sqlite3 tickets.db "select * from tickets;"
```

Set `LANGSMITH_TRACING=true` in `.env` to watch the whole run, including
each hop of the agent/tools loop, step by step in LangSmith.

## Tests

```bash
python -m pytest
```

Six tests, no API key or GPU required: routing logic, tool side effects,
entity extraction, graph compilation, and one full pipeline run with a
fake Whisper model, a fake translation/classification chain, and a fake
tool-calling LLM that requests `create_urgent_ticket` on the first turn
and gives a final answer on the second.

## Notes from getting this running

- `langchain-core` 1.0+ changed `StrOutputParser` to return a `str`
  subclass (`TextAccessor`) instead of a plain `str`. spaCy's tokenizer
  does a strict type check and rejects it, so `translate_node` casts the
  result with `str()` before it's used downstream.
- Groq's `openai/gpt-oss-20b` model is currently unreliable with
  LangChain's `with_structured_output()` (it sometimes calls a tool that
  was never registered in the request). Switched the default model to
  `llama-3.3-70b-versatile`, which handles structured/tool output
  correctly.

## Possible next steps

- Wrap `run.py` in a FastAPI endpoint (`POST /process-call`) to match
  the deployment style of my other projects.
- Add a `flag_for_review` tool for calls where entity extraction finds a
  named person but classification is `Open`, to catch cases worth a
  second look.
- Swap the keyword-based urgency scan in `extract_entities` for a second
  lightweight classifier and compare precision/recall against the
  current approach.
