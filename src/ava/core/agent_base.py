"""
Base Agent Class for Unified Agent Architecture
All agents inherit from this base class
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

import os

logger = logging.getLogger(__name__)

try:
    from langchain_huggingface import HuggingFaceEndpoint
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("langchain-huggingface not available, Hugging Face support disabled")


class AgentState(TypedDict):
    """State for individual agent execution"""
    input: str
    context: Dict[str, Any]
    tools: List[BaseTool]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base class for all AVA agents
    
    Provides:
    - LangChain tool integration
    - Hugging Face model support
    - State management
    - Error handling
    - Logging
    - Caching
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[BaseChatModel] = None,
        use_huggingface: bool = False,
        hf_model: Optional[str] = None,
        enable_learning: bool = True
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            description: Agent description
            tools: List of LangChain tools this agent can use
            llm: LLM instance (optional, will create if needed)
            use_huggingface: Whether to use Hugging Face models
            hf_model: Hugging Face model name
            enable_learning: Whether to enable learning system
        """
        self.name = name
        self.description = description
        self.tools = tools or []
        self.llm = llm
        self.use_huggingface = use_huggingface
        self.hf_model = hf_model or "meta-llama/Llama-3.1-8B-Instruct"
        self.enable_learning = enable_learning
        
        # Initialize learning system
        if enable_learning:
            try:
                from .agent_learning import AgentLearningSystem
                self.learning_system = AgentLearningSystem()
            except Exception as e:
                logger.warning(f"Learning system not available: {e}")
                self.learning_system = None
        else:
            self.learning_system = None
        
        # Initialize Hugging Face if requested
        if use_huggingface:
            self._init_huggingface()
        
        # Agent metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'capabilities': []
        }
        
        logger.info(f"Initialized agent: {self.name}")
    
    def _init_huggingface(self) -> None:
        """Initialize Hugging Face endpoint"""
        if not HUGGINGFACE_AVAILABLE:
            logger.warning("langchain-huggingface not installed, Hugging Face disabled")
            self.use_huggingface = False
            return
            
        try:
            hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not hf_api_key:
                logger.warning("HUGGINGFACE_API_KEY not set, Hugging Face disabled")
                self.use_huggingface = False
                return
            
            self.hf_llm = HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/{self.hf_model}",
                huggingface_api_key=hf_api_key,
                task="text-generation",
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 1000
                }
            )
            logger.info(f"Hugging Face initialized with model: {self.hf_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face: {e}")
            self.use_huggingface = False
    
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute agent logic (must be implemented by subclasses)
        
        Args:
            state: Agent state with input and context
            
        Returns:
            Updated state with result
        """
        pass
    
    async def execute_with_learning(self, state: AgentState) -> AgentState:
        """
        Execute agent with learning system integration
        
        This wraps the execute method to add:
        - Performance tracking
        - Execution logging
        - Memory storage
        """
        import time
        import uuid
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        input_text = state.get('input', '')
        user_id = state.get('context', {}).get('user_id')
        platform = state.get('context', {}).get('platform', 'web')
        
        try:
            # Execute agent
            result_state = await self.execute(state)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log execution if learning enabled
            if self.learning_system:
                self.learning_system.log_execution(
                    agent_name=self.name,
                    execution_id=execution_id,
                    input_text=input_text,
                    result=result_state.get('result', {}),
                    error=result_state.get('error'),
                    response_time_ms=response_time_ms,
                    user_id=user_id,
                    platform=platform
                )
            
            return result_state
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Log error if learning enabled
            if self.learning_system:
                self.learning_system.log_execution(
                    agent_name=self.name,
                    execution_id=execution_id,
                    input_text=input_text,
                    result={},
                    error=error_msg,
                    response_time_ms=response_time_ms,
                    user_id=user_id,
                    platform=platform
                )
            
            state['error'] = error_msg
            state['result'] = {'error': error_msg}
            return state
    
    def get_llm(self) -> Optional[BaseChatModel]:
        """Get LLM instance (Hugging Face or provided)"""
        if self.use_huggingface and hasattr(self, 'hf_llm'):
            return self.hf_llm
        return self.llm
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to this agent"""
        self.tools.append(tool)
        logger.info(f"Added tool {tool.name} to agent {self.name}")
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        return self.metadata.get('capabilities', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'tools': [tool.name for tool in self.tools],
            'capabilities': self.get_capabilities(),
            'metadata': self.metadata
        }

