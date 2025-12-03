"""
SpecAgent Registry - Central registration and lookup for all SpecAgents
"""

import logging
from typing import Dict, List, Optional, Type
from .base_spec_agent import BaseSpecAgent

logger = logging.getLogger(__name__)


class SpecAgentRegistry:
    """
    Central registry for all SpecAgents.

    Provides:
    - Agent registration
    - Agent lookup by feature name
    - Agent discovery by capability
    """

    _instance: Optional['SpecAgentRegistry'] = None
    _agents: Dict[str, BaseSpecAgent] = {}
    _agent_classes: Dict[str, Type[BaseSpecAgent]] = {}

    def __new__(cls) -> 'SpecAgentRegistry':
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._agents = {}
            cls._agent_classes = {}
        return cls._instance

    @classmethod
    def register_class(cls, feature_name: str, agent_class: Type[BaseSpecAgent]) -> None:
        """
        Register a SpecAgent class (not instance)

        Args:
            feature_name: Feature name (e.g., 'positions')
            agent_class: SpecAgent class to register
        """
        if feature_name in cls._agent_classes:
            logger.warning(f"Overwriting agent class for feature: {feature_name}")

        cls._agent_classes[feature_name] = agent_class
        logger.info(f"Registered SpecAgent class: {feature_name}")

    @classmethod
    def register(cls, agent: BaseSpecAgent) -> None:
        """
        Register a SpecAgent instance

        Args:
            agent: SpecAgent instance to register
        """
        feature_name = agent.feature_name

        if feature_name in cls._agents:
            logger.warning(f"Overwriting agent instance for feature: {feature_name}")

        cls._agents[feature_name] = agent
        logger.info(f"Registered SpecAgent instance: {feature_name}")

    @classmethod
    def get(cls, feature_name: str) -> Optional[BaseSpecAgent]:
        """
        Get a SpecAgent by feature name

        If only class is registered, instantiate it.
        """
        # Check for existing instance
        if feature_name in cls._agents:
            return cls._agents[feature_name]

        # Check for registered class
        if feature_name in cls._agent_classes:
            agent = cls._agent_classes[feature_name]()
            cls._agents[feature_name] = agent
            return agent

        return None

    @classmethod
    def get_all(cls) -> List[BaseSpecAgent]:
        """Get all registered SpecAgent instances"""
        # Instantiate any registered classes that haven't been instantiated
        for feature_name, agent_class in cls._agent_classes.items():
            if feature_name not in cls._agents:
                cls._agents[feature_name] = agent_class()

        return list(cls._agents.values())

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get all registered feature names"""
        all_features = set(cls._agents.keys()) | set(cls._agent_classes.keys())
        return sorted(all_features)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)"""
        cls._agents.clear()
        cls._agent_classes.clear()

    @classmethod
    def get_agent_count(cls) -> int:
        """Get count of registered agents"""
        return len(set(cls._agents.keys()) | set(cls._agent_classes.keys()))


def register_spec_agent(feature_name: str):
    """
    Decorator to register a SpecAgent class

    Usage:
        @register_spec_agent('positions')
        class PositionsSpecAgent(BaseSpecAgent):
            ...
    """
    def decorator(cls: Type[BaseSpecAgent]):
        SpecAgentRegistry.register_class(feature_name, cls)
        return cls
    return decorator
