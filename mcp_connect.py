import asyncio
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_discover import MCPToolDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTool:

    def __init__(
        self, name, description, input_schema, server_cmd, server_args
    ) -> None:
        self.name = name
        self._description = description
        self._input_schema = input_schema
        self._params = StdioServerParameters(command=server_cmd, args=server_args)

    async def run(self, args: dict):
        async with stdio_client(self._params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(self.name, args)
                return getattr(response, "content", str(response))


class MCPConnector:

    def __init__(self, config_file: str = None) -> None:
        self._discovery = MCPToolDiscovery(config_file)
        self._tools: list[MCPTool] = []
        self._load_all_tools()

    def _load_all_tools(self):

        async def _fetch():
            mcp_servers = self._discovery.list_servers()

            for name, info in mcp_servers.items():
                command = info.get("command")
                args = info.get("args", [])
                params = StdioServerParameters(command=command, args=args)

                try:
                    async with stdio_client(params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()

                            tools = (await session.list_tools()).tools

                            for tool in tools:
                                self._tools.append(
                                    MCPTool(
                                        name=tool.name,
                                        description=tool.description,
                                        input_schema=tool.inputSchema,
                                        server_args=args,
                                        server_cmd=command,
                                    )
                                )
                except Exception as e:
                    logger.error(
                        f"Error occurred while loading MCP tools\n Reason: {e}"
                    )

        asyncio.run(_fetch())

    def get_tools(self):
        return self._tools.copy()
