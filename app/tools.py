"""Tools exposed to the action agent (see graph.py's agent node).

The agent decides which one to call and with what arguments based on
the classification and transcript it's given. There's no hardcoded
"Urgent -> create_urgent_ticket" mapping in this file; that reasoning
lives entirely in the agent's system prompt in graph.py.
"""

from langchain_core.tools import tool

from . import db


@tool
def create_urgent_ticket(reason: str, customer_name: str = "unknown") -> str:
    """Escalate the call as a high-priority ticket for a human agent to act on immediately.

    Use for angry customers, safety/legal issues, or anything that cannot wait.
    """
    db.log_action("URGENT_TICKET", reason, customer_name)
    return f"Urgent ticket created for {customer_name}: {reason}"


@tool
def schedule_followup(reason: str, due_in_days: int = 2) -> str:
    """Schedule a non-urgent follow-up for a call that wasn't fully resolved.

    Use when the caller's issue needs more work but there's no immediate risk.
    """
    db.log_action("FOLLOWUP", reason)
    return f"Follow-up scheduled in {due_in_days} day(s): {reason}"


@tool
def close_ticket(summary: str) -> str:
    """Close out a call that was fully resolved during the conversation itself."""
    db.log_action("CLOSED", summary)
    return f"Ticket closed: {summary}"


TOOLS = [create_urgent_ticket, schedule_followup, close_ticket]
