"""CLI: run the full multi-agent call classifier on a single audio file.

Usage:
    python run.py sample/sample_calls/open_hindi.mp3
"""

import argparse
import json
import sys

from app.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent call classifier on a call recording.")
    parser.add_argument("audio_path", help="Path to a call recording (mp3/wav/m4a/...).")
    args = parser.parse_args()

    graph = build_graph()
    final_state = graph.invoke({"audio_path": args.audio_path, "messages": []})

    print("\n=== Transcript ===")
    print(final_state.get("transcript"))
    print("Detected language:", final_state.get("language"))

    if final_state.get("english_text"):
        print("\n=== English translation ===")
        print(final_state["english_text"])

    print("\n=== Extracted entities ===")
    print(json.dumps(final_state.get("entities"), indent=2))

    print("\n=== Classification ===")
    print(f"{final_state.get('classification')} - {final_state.get('reason')}")
    print(final_state.get("summary"))

    print("\n=== Agent trace ===")
    for message in final_state["messages"]:
        role = message.__class__.__name__.replace("Message", "")
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        if content:
            print(f"[{role}] {content}")
        if tool_calls:
            for call in tool_calls:
                print(f"[{role} -> tool call] {call['name']}({call['args']})")


if __name__ == "__main__":
    sys.exit(main())
