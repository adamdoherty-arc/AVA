"""
MCP Server for AVA Tools
Model Context Protocol server exposing AVA tools
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
from typing import List, Dict, Any
import logging
from src.ava.core import AVACore

logger = logging.getLogger(__name__)


class AVAMCPServer:
    """MCP Server for AVA tools"""

    def __init__(self, ava_core: AVACore):
        """
        Initialize MCP server

        Args:
            ava_core: AVA Core instance
        """
        self.ava = ava_core
        self.server = Server("ava-tools")
        
        # Register handlers
        self._register_handlers()
        
        logger.info("AVA MCP Server initialized")

    def _register_handlers(self) -> None:
        """Register MCP handlers"""
        self.server.list_tools()(self._list_tools)
        self.server.call_tool()(self._call_tool)

    async def _list_tools(self) -> List[Tool]:
        """
        List available tools

        Returns:
            List of Tool definitions
        """
        tools = []
        
        # Get tools from registry
        tool_list = self.ava.tool_registry.list_tools()
        
        for tool_info in tool_list:
            tool = Tool(
                name=tool_info["name"],
                description=tool_info["description"],
                inputSchema={
                    "type": "object",
                    "properties": self._get_tool_schema(tool_info["name"]),
                    "required": []
                }
            )
            tools.append(tool)
        
        return tools

    def _get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get JSON schema for a tool"""
        # Map tool names to their schemas
        schemas = {
            "query_database_tool": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute"
                }
            },
            "analyze_watchlist_tool": {
                "watchlist_name": {
                    "type": "string",
                    "description": "Name of watchlist to analyze"
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum profit score threshold (0-100)",
                    "default": 60.0
                }
            },
            "get_portfolio_status_tool": {},
            "create_task_tool": {
                "title": {
                    "type": "string",
                    "description": "Task title"
                },
                "description": {
                    "type": "string",
                    "description": "Task description"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium"
                }
            },
            "get_stock_price_tool": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "search_magnus_knowledge_tool": {
                "question": {
                    "type": "string",
                    "description": "Question about Magnus"
                }
            }
        }
        
        return schemas.get(tool_name, {})

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Call a tool

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            List of TextContent with results
        """
        try:
            # Get tool from registry
            tool = self.ava.tool_registry.get_tool(name)
            
            if not tool:
                return [TextContent(
                    type="text",
                    text=f"Tool '{name}' not found"
                )]
            
            # Execute tool
            result = tool.invoke(arguments)
            
            # Format result
            if isinstance(result, str):
                result_text = result
            else:
                import json
                result_text = json.dumps(result, indent=2, default=str)
            
            return [TextContent(
                type="text",
                text=result_text
            )]
            
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    async def run(self) -> None:
        """Run MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


# Standalone server runner
async def run_mcp_server():
    """Run MCP server standalone"""
    ava = AVACore()
    server = AVAMCPServer(ava)
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_mcp_server())

