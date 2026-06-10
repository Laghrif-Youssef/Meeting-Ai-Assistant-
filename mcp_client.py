from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)

from mcp import ClientSession

import asyncio


async def call_tool(tool_name, arguments, server):

    server_params = StdioServerParameters(
        command="python",
        args=["-m", server]
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            result = await session.call_tool(tool_name, arguments)

            return result.structuredContent