"""Env vars and lazy model/client construction, shared by every node."""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "groq:llama-3.3-70b-versatile")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")

_whisper_model = None
_nlp = None
_llm = None


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper

        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME).to(get_device())
    return _whisper_model


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_llm():
    global _llm
    if _llm is None:
        from langchain.chat_models import init_chat_model

        _llm = init_chat_model(GROQ_MODEL)
    return _llm
