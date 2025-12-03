"""
AVA Core - Unified Chatbot Implementation
Modern architecture with LangGraph, structured outputs, streaming, and MCP
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator, TypedDict
from datetime import datetime
import time

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain LLM imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Optional Google Generative AI - may have version compatibility issues
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except (ImportError, AttributeError) as e:
    logging.warning(f"Google Generative AI import failed: {e}. Google LLM will be unavailable.")
    ChatGoogleGenerativeAI = None

# Local imports
from .models import IntentResult, MessageResponse, ConversationState, AVAConfig
from .state_manager import AVAStateManager
from .tool_registry import ToolRegistry
from src.rag.rag_service import RAGService
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# LangGraph State Definition
class AVAState(TypedDict):
    """State for LangGraph workflow"""
    messages: List[BaseMessage]
    user_id: str
    platform: str
    intent: Optional[IntentResult]
    rag_results: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    response: Optional[str]
    context: Dict[str, Any]


class AVACore:
    """
    Unified AVA Core Implementation
    
    Features:
    - LangGraph-based state machine workflows
    - Structured outputs with Pydantic
    - Streaming responses
    - RAG integration (default for knowledge queries)
    - Tool execution
    - Unified state management
    - Multi-platform support
    """

    def __init__(self, config: Optional[AVAConfig] = None):
        """
        Initialize AVA Core

        Args:
            config: AVA configuration (uses defaults if None)
        """
        self.config = config or AVAConfig()
        self.state_manager = AVAStateManager()
        self.tool_registry = ToolRegistry()
        self.rag_service = RAGService()
        self.llm_service = LLMService()

        # Initialize LLM for structured outputs
        self._init_llm()

        # Register default tools
        self._register_default_tools()

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

        logger.info("AVACore initialized successfully")

    def _init_llm(self) -> None:
        """Initialize LLM for structured outputs and function calling"""
        try:
            # Try Groq first (free tier)
            groq_key = os.getenv('GROQ_API_KEY')
            if groq_key:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",  # Updated to latest model
                    temperature=self.config.temperature,
                    groq_api_key=groq_key
                )
                # Bind tools for function calling
                tools = self.tool_registry.get_all_tools()
                if tools:
                    self.llm_with_tools = self.llm.bind_tools(tools)
                else:
                    self.llm_with_tools = self.llm
                logger.info("Using Groq LLM with function calling")
                return
        except Exception as e:
            logger.warning(f"Groq not available: {e}")

        try:
            # Fallback to OpenAI (better function calling support)
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=self.config.temperature
                )
                # Bind tools for function calling
                tools = self.tool_registry.get_all_tools()
                if tools:
                    self.llm_with_tools = self.llm.bind_tools(tools)
                else:
                    self.llm_with_tools = self.llm
                logger.info("Using OpenAI LLM with function calling")
                return
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")

        # Fallback to LLMService
        self.llm = None
        self.llm_with_tools = None
        logger.warning("Using LLMService fallback (no structured outputs or function calling)")

    def _register_default_tools(self) -> None:
        """Register default tools"""
        from src.ava.core.tools import (
            query_database_tool,
            analyze_watchlist_tool,
            get_portfolio_status_tool,
            create_task_tool,
            get_stock_price_tool,
            search_magnus_knowledge_tool,
            # Sports betting tools
            get_kalshi_markets_tool,
            get_live_games_tool,
            get_game_watchlist_tool,
            get_betting_opportunities_tool,
            # Trading tools
            get_positions_tool,
            get_trading_opportunities_tool,
            get_trade_history_tool,
            # Task management tools
            get_tasks_tool,
            # Xtrades tools
            get_xtrades_profiles_tool,
            get_xtrades_trades_tool
        )

        # Core tools
        self.tool_registry.register_tool(query_database_tool)
        self.tool_registry.register_tool(analyze_watchlist_tool)
        self.tool_registry.register_tool(get_portfolio_status_tool)
        self.tool_registry.register_tool(create_task_tool)
        self.tool_registry.register_tool(get_stock_price_tool)
        self.tool_registry.register_tool(search_magnus_knowledge_tool)
        
        # Sports betting tools
        self.tool_registry.register_tool(get_kalshi_markets_tool)
        self.tool_registry.register_tool(get_live_games_tool)
        self.tool_registry.register_tool(get_game_watchlist_tool)
        self.tool_registry.register_tool(get_betting_opportunities_tool)
        
        # Trading tools
        self.tool_registry.register_tool(get_positions_tool)
        self.tool_registry.register_tool(get_trading_opportunities_tool)
        self.tool_registry.register_tool(get_trade_history_tool)
        
        # Task management tools
        self.tool_registry.register_tool(get_tasks_tool)
        
        # Xtrades tools
        self.tool_registry.register_tool(get_xtrades_profiles_tool)
        self.tool_registry.register_tool(get_xtrades_trades_tool)

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AVAState)

        # Add nodes
        workflow.add_node("detect_intent", self._detect_intent_node)
        workflow.add_node("query_rag", self._query_rag_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Set entry point
        workflow.set_entry_point("detect_intent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "detect_intent",
            self._route_after_intent,
            {
                "rag": "query_rag",
                "tools": "execute_tools",
                "direct": "generate_response"
            }
        )

        # Add edges
        workflow.add_edge("query_rag", "generate_response")
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("generate_response", END)

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _detect_intent_node(self, state: AVAState) -> AVAState:
        """Detect user intent with structured output"""
        last_message = state["messages"][-1].content if state["messages"] else ""

        try:
            if self.llm:
                # Use structured output
                parser = PydanticOutputParser(pydantic_object=IntentResult)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self._get_intent_prompt()),
                    ("human", "User query: {query}\n\n{format_instructions}")
                ])

                chain = prompt | self.llm | parser
                intent_result: IntentResult = chain.invoke({
                    "query": last_message,
                    "format_instructions": parser.get_format_instructions()
                })
            else:
                # Fallback to LLMService
                response = self.llm_service.generate(
                    prompt=self._get_intent_prompt() + f"\n\nUser query: {last_message}",
                    max_tokens=300,
                    temperature=0.1
                )
                intent_result = self._parse_intent_fallback(response['text'])

            state["intent"] = intent_result
            logger.info(f"Intent detected: {intent_result.intent} (confidence: {intent_result.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Fallback intent
            state["intent"] = IntentResult(
                intent="unknown",
                confidence=0.5,
                needs_rag=True
            )

        return state

    def _query_rag_node(self, state: AVAState) -> AVAState:
        """Query RAG knowledge base"""
        if not self.config.enable_rag:
            return state

        last_message = state["messages"][-1].content if state["messages"] else ""

        try:
            # RAG service uses 'question' parameter
            rag_result = self.rag_service.query(
                question=last_message,
                use_cache=True
            )

            state["rag_results"] = {
                "answer": rag_result.answer,
                "sources": rag_result.sources,
                "confidence": rag_result.confidence
            }

            logger.info(f"RAG query completed (confidence: {rag_result.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            state["rag_results"] = None

        return state

    def _execute_tools_node(self, state: AVAState) -> AVAState:
        """Execute required tools (supports both manual and function calling)"""
        intent = state.get("intent")
        tool_results = []

        # Check for function calls in messages (from LLM with function calling)
        last_message = state["messages"][-1] if state["messages"] else None
        
        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Function calling mode - execute tools from LLM
            for tool_call in last_message.tool_calls:
                try:
                    tool_name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
                    tool_args = tool_call.get("args", tool_call.get("function", {}).get("arguments", {}))
                    
                    if isinstance(tool_args, str):
                        import json
                        tool_args = json.loads(tool_args)
                    
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        result = tool.invoke(tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "result": result,
                            "success": True,
                            "tool_call_id": tool_call.get("id", "")
                        })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })
        elif intent and intent.needs_tools:
            # Manual tool execution mode
            for tool_name in intent.needs_tools:
                try:
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        # Extract arguments from intent entities
                        arguments = intent.entities.get(tool_name, {})
                        result = tool.invoke(arguments)
                        tool_results.append({
                            "tool": tool_name,
                            "result": result,
                            "success": True
                        })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })

        state["tool_results"] = tool_results
        return state

    def _generate_response_node(self, state: AVAState) -> AVAState:
        """Generate final response"""
        last_message = state["messages"][-1].content if state["messages"] else ""
        intent = state.get("intent")
        rag_results = state.get("rag_results")
        tool_results = state.get("tool_results", [])

        # Build response prompt
        prompt_parts = [
            "You are AVA, an expert AI trading assistant.",
            f"User query: {last_message}",
        ]

        if intent:
            prompt_parts.append(f"Intent: {intent.intent} (confidence: {intent.confidence:.2f})")

        if rag_results:
            prompt_parts.append(f"Knowledge base answer: {rag_results.get('answer', '')}")
            prompt_parts.append(f"Sources: {rag_results.get('sources', [])}")

        if tool_results:
            prompt_parts.append("Tool results:")
            for tr in tool_results:
                prompt_parts.append(f"- {tr['tool']}: {tr.get('result', 'N/A')}")

        prompt = "\n".join(prompt_parts)

        try:
            if self.llm_with_tools:
                # Use LLM with function calling if tools are available
                response = self.llm_with_tools.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            elif self.llm:
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                response_text = llm_response['text']

            state["response"] = response_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["response"] = "I encountered an error processing your request. Please try again."

        return state

    def _route_after_intent(self, state: AVAState) -> str:
        """Route to appropriate node after intent detection"""
        intent = state.get("intent")

        if not intent:
            return "direct"

        if intent.needs_rag and self.config.enable_rag:
            return "rag"

        if intent.needs_tools:
            return "tools"

        return "direct"

    async def process_message(
        self,
        message: str,
        user_id: str,
        platform: str = "web"
    ) -> AsyncIterator[str]:
        """
        Process message with streaming response

        Args:
            message: User message
            user_id: User identifier
            platform: Platform (web, telegram, api)

        Yields:
            Response chunks as they're generated
        """
        start_time = time.time()

        # Get or create conversation state
        conversation_state = self.state_manager.get_state(user_id, platform)

        # Build initial state
        config = {"configurable": {"thread_id": f"{platform}:{user_id}"}}

        initial_state: AVAState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "platform": platform,
            "intent": None,
            "rag_results": None,
            "tool_results": [],
            "response": None,
            "context": conversation_state.context
        }

        try:
            # Stream workflow execution
            async for chunk in self.workflow.astream(initial_state, config=config):
                # Yield intermediate updates
                if "generate_response" in chunk:
                    response = chunk["generate_response"].get("response")
                    if response:
                        if self.config.enable_streaming:
                            # Stream response word by word
                            words = response.split()
                            for word in words:
                                yield word + " "
                                await asyncio.sleep(0.05)  # Small delay for streaming effect
                        else:
                            yield response

            # Get final state
            final_state = await self.workflow.ainvoke(initial_state, config=config)

            # Update conversation state
            self.state_manager.update_state(
                user_id=user_id,
                platform=platform,
                messages=[{"role": "user", "content": message}],
                context={"last_intent": final_state.get("intent")}
            )

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Message processed in {latency_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield f"I encountered an error: {str(e)}"

    def process_message_sync(
        self,
        message: str,
        user_id: str,
        platform: str = "web"
    ) -> MessageResponse:
        """
        Process message synchronously (non-streaming)

        Args:
            message: User message
            user_id: User identifier
            platform: Platform identifier

        Returns:
            MessageResponse with full response
        """
        start_time = time.time()

        # Use direct workflow invocation for sync processing
        config = {"configurable": {"thread_id": f"{platform}:{user_id}"}}
        initial_state: AVAState = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "platform": platform,
            "intent": None,
            "rag_results": None,
            "tool_results": [],
            "response": None,
            "context": {}
        }
        
        try:
            # Try to get running loop
            try:
                loop = asyncio.get_running_loop()
                # Loop is running, use nest_asyncio or direct invocation
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    final_state = loop.run_until_complete(
                        self.workflow.ainvoke(initial_state, config=config)
                    )
                except (ImportError, RuntimeError):
                    # Direct synchronous execution
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self.workflow.ainvoke(initial_state, config=config)
                            )
                        )
                        final_state = future.result(timeout=30)
            except RuntimeError:
                # No loop running, create new one
                final_state = asyncio.run(
                    self.workflow.ainvoke(initial_state, config=config)
                )

            full_response = final_state.get("response", "Response generated")
            latency_ms = (time.time() - start_time) * 1000

            return MessageResponse(
                content=full_response,
                intent="unknown",  # Will be set from state
                confidence=0.8,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Error in sync processing: {e}")
            return MessageResponse(
                content=f"I encountered an error: {str(e)}",
                intent="error",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000
            )

    def _get_intent_prompt(self) -> str:
        """Get prompt for intent detection"""
        return """You are an intent classifier for AVA, a trading assistant.

Analyze the user query and determine:
1. Intent (portfolio, positions, opportunities, tradingview, xtrades, tasks, status, help, unknown)
2. Confidence (0.0-1.0)
3. Entities (tickers, dates, numbers, etc.)
4. Whether RAG knowledge base is needed
5. Which tools are needed (if any)

Available intents:
- portfolio: Portfolio balance, performance, overview
- positions: Active options positions, trades
- opportunities: Trading opportunities, CSP plays
- tradingview: TradingView watchlists, charts
- xtrades: Xtrades following, signals
- tasks: Task management, what AVA is working on
- status: System status, health check
- help: Help, commands, what can you do
- unknown: Doesn't match any intent

Respond with structured JSON matching the IntentResult schema."""

    def _parse_intent_fallback(self, text: str) -> IntentResult:
        """Fallback intent parsing from text"""
        # Simple keyword-based parsing
        text_lower = text.lower()

        intent_map = {
            "portfolio": ["portfolio", "balance", "account", "money"],
            "positions": ["position", "trade", "option", "holding"],
            "opportunities": ["opportunity", "play", "csp", "trade idea"],
            "tradingview": ["tradingview", "watchlist", "chart"],
            "xtrades": ["xtrades", "follow", "trader"],
            "tasks": ["task", "working", "doing"],
            "status": ["status", "online", "health"],
            "help": ["help", "command", "what can"]
        }

        for intent, keywords in intent_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return IntentResult(
                    intent=intent,
                    confidence=0.7,
                    needs_rag="what" in text_lower or "how" in text_lower or "explain" in text_lower
                )

        return IntentResult(intent="unknown", confidence=0.5, needs_rag=True)


import os

