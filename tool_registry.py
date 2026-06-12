"""
Tool Registry — manages MCP server connections and tool execution.

Key fix vs original: instead of opening + closing a new subprocess for every
single tool call (very slow, fragile), we keep one persistent MCP session per
server alive for the duration of an agent run via context managers.
"""
import asyncio
from contextlib import asynccontextmanager

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


# ─────────────────────────────────────────────
# Registry: { tool_name -> { server, schema } }
# ─────────────────────────────────────────────
TOOL_REGISTRY: dict[str, dict] = {}

SERVERS = [
    "Gmail_MCP_Server",
    "Calendar_MCP_Server",
    "Audio_Transcription_MCP_Server"
]


async def _list_tools_for_server(server_name: str) -> list:
    """Open a short-lived session just to discover tools at startup."""
    params = StdioServerParameters(command="python", args=["-m", server_name])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def init_registry():
    """Discover all tools from all MCP servers and populate TOOL_REGISTRY."""
    for server_name in SERVERS:
        tools = await _list_tools_for_server(server_name)
        for t in tools:
            TOOL_REGISTRY[t.name] = {
                "server": server_name,
                "schema": t.inputSchema,
                "description": t.description or "",
            }
    print(f"[registry] Loaded {len(TOOL_REGISTRY)} tools: {list(TOOL_REGISTRY.keys())}")


# ─────────────────────────────────────────────
# Persistent session pool (one per server per run)
# ─────────────────────────────────────────────

_active_sessions: dict[str, ClientSession] = {}
_active_cm_stack = []          # keep context managers alive


async def open_sessions():
    """
    Open one persistent MCP session per server.
    Call once before the agent run starts, close_sessions() when done.
    """
    for server_name in SERVERS:
        params = StdioServerParameters(command="python", args=["-m", server_name])
        cm = stdio_client(params)
        streams = await cm.__aenter__()
        _active_cm_stack.append(cm)

        session_cm = ClientSession(streams[0], streams[1])
        session = await session_cm.__aenter__()
        await session.initialize()

        _active_sessions[server_name] = session
        _active_cm_stack.append(session_cm)

    print(f"[registry] Persistent sessions open for: {list(_active_sessions.keys())}")


async def close_sessions():
    """Close all persistent MCP sessions."""
    for cm in reversed(_active_cm_stack):
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass
    _active_sessions.clear()
    _active_cm_stack.clear()


async def execute_tool(tool_name: str, arguments: dict):
    """
    Execute a tool via its MCP server.
    Reuses the persistent session if available, falls back to a fresh one.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: '{tool_name}'. "
                         f"Available: {list(TOOL_REGISTRY.keys())}")

    server_name = TOOL_REGISTRY[tool_name]["server"]

    # Use persistent session if available
    if server_name in _active_sessions:
        session = _active_sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result.content

    # Fallback: open a fresh session (slower but safe)
    params = StdioServerParameters(command="python", args=["-m", server_name])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content