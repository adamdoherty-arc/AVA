"""
Agent Invoker Tool - Allows AVA to invoke any registered agent
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.ava.core.agent_initializer import get_registry
from src.ava.core.agent_learning import AgentLearningSystem

logger = logging.getLogger(__name__)


class AgentInvokerInput(BaseModel):
    """Input for agent invoker tool"""
    agent_name: str = Field(description="Name of the agent to invoke (e.g., 'kalshi_markets_agent', 'options_analysis_agent')")
    input: str = Field(description="Input/question for the agent")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional context for the agent")


class AgentInvokerTool(BaseTool):
    """Tool that allows AVA to invoke any registered agent"""

    name = "invoke_agent"
    description = """
    Invoke any registered agent in the system. Use this when you need specialized functionality.

    Available agents include:
    - Trading agents: market_data_agent, options_analysis_agent, strategy_agent, risk_management_agent
    - Sports agents: kalshi_markets_agent, sports_betting_agent, nfl_markets_agent, game_analysis_agent
    - Analysis agents: fundamental_analysis_agent, technical_analysis_agent, sentiment_analysis_agent
    - Monitoring agents: watchlist_monitor_agent, xtrades_monitor_agent, alert_agent
    - Research agents: knowledge_agent, research_agent, documentation_agent
    - Management agents: task_management_agent, position_management_agent
    - Code agents: code_recommendation_agent, claude_code_controller_agent, qa_agent

    Example usage:
    - "Invoke kalshi_markets_agent with 'Get NFL markets for this weekend'"
    - "Use options_analysis_agent to analyze TSLA options"
    - "Ask game_analysis_agent about Dallas Cowboys vs Las Vegas Raiders"
    """
    args_schema = AgentInvokerInput

    def __init__(self) -> None:
        super().__init__()
        self.registry = get_registry()
        self.learning_system = AgentLearningSystem()

    def _run(self, agent_name: str, input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Invoke an agent"""
        try:
            from datetime import datetime

            # Get the agent
            agent = self.registry.get_agent(agent_name)

            if not agent:
                available_agents = self.registry.list_agent_names()
                return f"❌ Agent '{agent_name}' not found. Available agents: {', '.join(available_agents[:10])}..."

            # Prepare context
            if context is None:
                context = {}

            context.update({
                "invoked_by": "ava",
                "platform": "ava_chatbot",
                "timestamp": datetime.now().isoformat()
            })

            # Execute the agent
            logger.info(f"AVA invoking agent: {agent_name} with input: {input[:100]}")

            start_time = datetime.now()
            response = agent.execute(
                input=input,
                context=context
            )
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds() * 1000

            # Log execution
            try:
                self.learning_system.log_execution(
                    agent_name=agent_name,
                    execution_id=f"ava_{int(start_time.timestamp())}",
                    input_text=input,
                    output_text=str(response)[:1000],
                    response_time_ms=execution_time,
                    success=True,
                    platform="ava_chatbot"
                )
            except Exception as log_error:
                logger.warning(f"Failed to log agent execution: {log_error}")

            # Format response
            if isinstance(response, dict):
                if 'output' in response:
                    result = response['output']
                elif 'result' in response:
                    result = response['result']
                elif 'response' in response:
                    result = response['response']
                else:
                    result = str(response)
            else:
                result = str(response)

            return f"✅ **{agent.name}** ({execution_time:.0f}ms):\n\n{result}"

        except Exception as e:
            logger.error(f"Error invoking agent {agent_name}: {e}")

            # Log failed execution
            try:
                self.learning_system.log_execution(
                    agent_name=agent_name,
                    execution_id=f"ava_error_{int(datetime.now().timestamp())}",
                    input_text=input,
                    output_text="",
                    response_time_ms=0,
                    success=False,
                    error=str(e),
                    platform="ava_chatbot"
                )
            except:
                pass

            return f"❌ Error invoking {agent_name}: {str(e)}"

    async def _arun(self, agent_name: str, input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Async version"""
        return self._run(agent_name, input, context)


class ListAgentsTool(BaseTool):
    """Tool to list all available agents"""

    name = "list_agents"
    description = "List all available agents in the system with their descriptions and capabilities"

    def __init__(self) -> None:
        super().__init__()
        self.registry = get_registry()

    def _run(self) -> str:
        """List all agents"""
        try:
            from src.ava.core.agent_initializer import ensure_agents_initialized

            # Ensure agents are initialized
            ensure_agents_initialized()

            agents = self.registry.get_all_agents()

            if not agents:
                return "❌ No agents found in registry"

            # Categorize agents
            categories = {
                "Trading": [],
                "Sports Betting": [],
                "Analysis": [],
                "Monitoring": [],
                "Research": [],
                "Management": [],
                "Code Development": []
            }

            keywords_map = {
                "Trading": ["market_data", "options", "strategy", "risk", "portfolio", "earnings", "premium"],
                "Sports Betting": ["kalshi", "sports", "nfl", "game", "odds", "betting"],
                "Analysis": ["fundamental", "technical", "sentiment", "supply", "sector", "options_flow"],
                "Monitoring": ["watchlist_monitor", "xtrades", "alert", "price_action"],
                "Research": ["knowledge", "research", "documentation"],
                "Management": ["task", "position", "settings"],
                "Code Development": ["code_recommendation", "claude_code", "qa"]
            }

            # Categorize each agent
            for agent in agents:
                if not agent:
                    continue

                categorized = False
                for category, keywords in keywords_map.items():
                    if any(kw in agent.name.lower() for kw in keywords):
                        categories[category].append(f"  • **{agent.name}**: {agent.description}")
                        categorized = True
                        break

                if not categorized:
                    if "Other" not in categories:
                        categories["Other"] = []
                    categories["Other"].append(f"  • **{agent.name}**: {agent.description}")

            # Format output
            result = f"📋 **Available Agents** ({len(agents)} total):\n\n"

            for category, agent_list in categories.items():
                if agent_list:
                    result += f"**{category}** ({len(agent_list)}):\n"
                    result += "\n".join(agent_list)
                    result += "\n\n"

            result += "💡 Use `invoke_agent` to execute any agent with specific input"

            return result

        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return f"❌ Error listing agents: {str(e)}"

    async def _arun(self) -> str:
        """Async version"""
        return self._run()


# Export tools
__all__ = ['AgentInvokerTool', 'ListAgentsTool']
