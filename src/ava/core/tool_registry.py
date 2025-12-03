"""
Tool Registry for AVA
Centralized tool registration and execution
"""

from typing import Dict, Callable, Any, Optional, List
import logging
from .models import ToolResult
import time

logger = logging.getLogger(__name__)

# Import LangChain tools
try:
    from langchain_core.tools import tool, BaseTool
except ImportError:
    # Fallback if not available
    BaseTool = object
    def tool(func):
        return func


class ToolRegistry:
    """Centralized tool registry for AVA"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("ToolRegistry initialized")

    def register_tool(
        self,
        tool_func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Register a tool function

        Args:
            tool_func: Tool function (can be already decorated with @tool)
            name: Optional tool name (defaults to function name)
            description: Optional tool description
        """
        # Check if already a LangChain tool
        if isinstance(tool_func, BaseTool):
            langchain_tool = tool_func
            tool_name = name or langchain_tool.name
        elif hasattr(tool_func, 'name') and hasattr(tool_func, 'invoke'):
            # Already a tool instance
            langchain_tool = tool_func
            tool_name = name or langchain_tool.name
        else:
            # Create LangChain tool
            langchain_tool = tool(tool_func)
            tool_name = name or getattr(
                tool_func, '__name__', langchain_tool.name
            )

        self._tools[tool_name] = langchain_tool

        # Get description
        tool_desc = description
        if not tool_desc:
            if hasattr(langchain_tool, 'description'):
                tool_desc = langchain_tool.description
            else:
                tool_desc = tool_func.__doc__ or ""

        self._tool_metadata[tool_name] = {
            "name": tool_name,
            "description": tool_desc,
            "function": tool_func
        }

        logger.info("Registered tool: %s", tool_name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution result
        """
        start_time = time.time()

        try:
            tool = self._tools.get(tool_name)

            if not tool:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Tool '{tool_name}' not found",
                    execution_time_ms=0.0
                )

            # Execute tool
            result = tool.invoke(arguments)
            execution_time = (time.time() - start_time) * 1000

            logger.info(
                "Executed tool '%s' in %.2fms", tool_name, execution_time
            )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Error executing tool '%s': %s", tool_name, e)

            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata"""
        return list(self._tool_metadata.values())

