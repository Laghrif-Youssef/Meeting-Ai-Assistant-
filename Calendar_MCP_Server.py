from fastmcp import FastMCP
from typing import Optional, Union
import json


from calendar_api import (
    create_event,
    get_upcoming_events,
    delete_event
)

mcp = FastMCP(
    name="Calendar_MCP_Server"
)

def _parse_attendees(attendees: Optional[Union[list, str]]) -> Optional[list[str]]:
    """
    Normalise attendees peu importe ce que l'LLM envoie :
      - None / []          → None
      - ["a@b.com"]        → ["a@b.com"]        (déjà une liste)
      - "a@b.com"          → ["a@b.com"]         (email brut)
      - '["a@b.com"]'      → ["a@b.com"]         (JSON string)
      - "a@b.com,b@c.com"  → ["a@b.com","b@c.com"] (CSV)
    """
    if not attendees:
        return None

    # Déjà une liste propre
    if isinstance(attendees, list):
        return [e.strip() for e in attendees if e.strip()] or None

    # String → essayer JSON d'abord
    s = attendees.strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [e.strip() for e in parsed if e.strip()] or None
        except json.JSONDecodeError:
            pass

    # CSV ou email unique
    return [e.strip() for e in s.split(",") if e.strip()] or None

@mcp.tool()
def create_event_tool(
    title: str,
    start_time: str,
    end_time: str,
    attendees: Optional[list[str]] = None   # ← liste d'emails
):
    return create_event(
        title,
        start_time,
        end_time,
        _parse_attendees(attendees)
    )

@mcp.tool()
def get_upcoming_events_tool(
    n: int
):
    return get_upcoming_events(n)


@mcp.tool()
def delete_event_tool(
    event_id: str
):
    return delete_event(event_id)


if __name__ == "__main__":
    mcp.run()
