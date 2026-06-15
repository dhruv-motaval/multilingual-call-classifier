# Multilingual Customer Support Call Classifier

A POC AI pipeline that automatically transcribes, translates, and classifies customer support calls in **Hindi and English** — built for a local company to help triage incoming support tickets without manual review.

---

## What It Does

Takes a raw audio call recording → outputs a structured classification with reason and summary.

```
Audio (Hindi/English)
        ↓
   Whisper STT  → detects language automatically
        ↓
  Translation   → non-English transcripts translated to English via Groq
        ↓
  spaCy NER     → extracts person names, urgent keywords
        ↓
  LLM Classify  → labels call as Open / Closed / Urgent + reason + summary
```

**Example Output:**
```
classification: Open
Reason: Device functionality issue that requires follow-up support.
Summary: User reports motion sensor not detecting movement. App settings did not help. User requests an engineer visit or a callback.
```

---

## Stack

| Component | Tool |
|---|---|
| Speech-to-Text | OpenAI Whisper (medium) |
| Translation + Classification | LangChain + Groq API |
| NLP / Entity Extraction | spaCy (en_core_web_sm) |
| LLM | Groq (via LangChain) |

---

## Classification Labels

| Label | Meaning |
|---|---|
| `Open` | Issue unresolved, needs follow-up |
| `Closed` | Issue resolved, no action needed |
| `Urgent` | Angry customer / safety / legal concern |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/dhruv-motaval/multilingual-call-classifier
cd multilingual-call-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> FFmpeg must be installed on your system for Whisper audio processing.  
> Install: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)

**3. Set up environment**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**4. Run**

Open `main.ipynb` in Jupyter and run all cells.  
Point `transcribe_audio()` to any `.mp3` or `.wav` file.

---

## Sample Calls

Six sample audio files are included in `sample/sample_calls/` covering all label combinations:

```
open_hindi.mp3
open_english.mp3
urgent_hindi.mp3
urgent_english.mp3
closed_hindi.mp3
closed_english.mp3
```

---

## Real-World Context

Built as a POC for a local IoT/hardware company to automate first-pass triage of inbound customer support calls. The system handles code-mixed Hindi-English speech (common in Indian customer calls) without any manual language selection.

---

## Author

**Dhruv Motaval**  
[GitHub](https://github.com/dhruv-motaval) · [LinkedIn](https://linkedin.com/in/dhruv-motaval) · [HuggingFace](https://huggingface.co/MotavalD/spaces)
