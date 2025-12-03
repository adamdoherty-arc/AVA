"""
Magnus AI Spec Agent System

Intelligent agents that deeply understand features and find real issues
through runtime testing (HTTP, Playwright, Database).
"""

from .base_spec_agent import (
    BaseSpecAgent,
    Issue,
    IssueSeverity,
    TestResult,
    # Schema validation
    SchemaType,
    SchemaField,
    ResponseSchema,
    SchemaValidator,
)
from .spec_agent_registry import SpecAgentRegistry
from .spec_orchestrator import SpecAgentOrchestrator, get_orchestrator
from .spec_agent_supervisor import SpecAgentSupervisor, SPEC_AGENT_TOOLS
from .dependency_graph import (
    FeatureDependencyGraph,
    get_dependency_graph,
    Dependency,
    DependencyType,
)

__all__ = [
    # Base classes
    'BaseSpecAgent',
    'Issue',
    'IssueSeverity',
    'TestResult',
    # Schema validation
    'SchemaType',
    'SchemaField',
    'ResponseSchema',
    'SchemaValidator',
    # Orchestration
    'SpecAgentRegistry',
    'SpecAgentOrchestrator',
    'get_orchestrator',
    'SpecAgentSupervisor',
    'SPEC_AGENT_TOOLS',
    # Dependency Graph
    'FeatureDependencyGraph',
    'get_dependency_graph',
    'Dependency',
    'DependencyType',
]
