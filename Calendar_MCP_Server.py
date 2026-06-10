from fastmcp import FastMCP

from calendar_api import (
    create_event,
    get_upcoming_events,
    delete_event
)

mcp = FastMCP(
    name="Calendar_MCP_Server"
)


@mcp.tool()
def create_event_tool(
    title: str,
    start_time: str,
    end_time: str
):
    return create_event(
        title,
        start_time,
        end_time
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