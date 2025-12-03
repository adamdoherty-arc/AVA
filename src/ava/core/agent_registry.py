"""
Agent Registry - Central registry for all AVA agents
Manages agent registration, routing, and lifecycle
"""

import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from .agent_base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for all agents
    
    Features:
    - Agent registration and discovery
    - Capability-based routing
    - Agent lifecycle management
    - Agent metadata tracking
    """
    
    def __init__(self) -> None:
        """Initialize agent registry"""
        self._agents: Dict[str, BaseAgent] = {}
        self._capabilities: Dict[str, List[str]] = {}  # capability -> [agent_names]
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AgentRegistry initialized")
    
    def register(self, agent: BaseAgent, capabilities: Optional[List[str]] = None):
        """
        Register an agent
        
        Args:
            agent: Agent instance
            capabilities: List of capabilities this agent provides
        """
        if agent.name in self._agents:
            logger.warning(f"Agent {agent.name} already registered, overwriting")
        
        self._agents[agent.name] = agent
        
        # Register capabilities
        if capabilities:
            for capability in capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = []
                if agent.name not in self._capabilities[capability]:
                    self._capabilities[capability].append(agent.name)
        else:
            # Use agent's own capabilities
            agent_caps = agent.get_capabilities()
            for capability in agent_caps:
                if capability not in self._capabilities:
                    self._capabilities[capability] = []
                if agent.name not in self._capabilities[capability]:
                    self._capabilities[capability].append(agent.name)
        
        # Store metadata
        self._metadata[agent.name] = {
            'registered_at': datetime.now().isoformat(),
            'capabilities': capabilities or agent.get_capabilities(),
            'description': agent.description,
            'tools': [tool.name for tool in agent.tools]
        }
        
        logger.info(f"Registered agent: {agent.name} with capabilities: {capabilities}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self._agents.get(name)
    
    def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Find all agents with a specific capability"""
        agent_names = self._capabilities.get(capability, [])
        return [self._agents[name] for name in agent_names if name in self._agents]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def get_all_capabilities(self) -> List[str]:
        """Get all registered capabilities"""
        return list(self._capabilities.keys())
    
    def route_request(self, query: str, required_capabilities: List[str]) -> List[BaseAgent]:
        """
        Route request to appropriate agents based on capabilities
        
        Args:
            query: User query
            required_capabilities: List of required capabilities
            
        Returns:
            List of agents that can handle the request
        """
        matching_agents = []
        
        for capability in required_capabilities:
            agents = self.find_agents_by_capability(capability)
            matching_agents.extend(agents)
        
        # Remove duplicates
        seen = set()
        unique_agents = []
        for agent in matching_agents:
            if agent.name not in seen:
                seen.add(agent.name)
                unique_agents.append(agent)
        
        return unique_agents
    
    def get_agent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        if name not in self._metadata:
            return None
        
        agent = self._agents.get(name)
        info = self._metadata[name].copy()
        
        if agent:
            info['agent_dict'] = agent.to_dict()
        
        return info
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents with their information"""
        return [
            self.get_agent_info(name)
            for name in self._agents.keys()
        ]
    
    def list_agent_names(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())

