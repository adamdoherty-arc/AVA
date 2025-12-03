"""
Cross-Feature Dependency Graph

Defines relationships between features to:
- Determine test execution order
- Identify cascade failures
- Track cross-feature impacts
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies between features"""
    REQUIRES = "requires"  # Feature A requires Feature B to work
    DATA_FROM = "data_from"  # Feature A uses data from Feature B
    SHARES_API = "shares_api"  # Features share API endpoints
    SHARES_DB = "shares_db"  # Features share database tables
    UI_EMBEDS = "ui_embeds"  # Feature A embeds UI from Feature B


@dataclass
class Dependency:
    """A dependency between two features"""
    from_feature: str
    to_feature: str
    dependency_type: DependencyType
    description: str = ""
    critical: bool = False  # If True, failure in dependency blocks this feature

    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_feature': self.from_feature,
            'to_feature': self.to_feature,
            'type': self.dependency_type.value,
            'description': self.description,
            'critical': self.critical,
        }


@dataclass
class FeatureNode:
    """A feature in the dependency graph"""
    name: str
    tier: int  # 1, 2, or 3
    dependencies: List[Dependency] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # Features that depend on this one


class FeatureDependencyGraph:
    """
    Manages cross-feature dependencies for the Magnus trading platform.

    Features and their relationships:
    - Tier 1 (Critical): positions, dashboard, premium-scanner, options-analysis, game-cards
    - Tier 2 (Important): ava-chatbot, kalshi-markets, earnings-calendar, xtrades-watchlists, dte-scanner
    - Tier 3 (Additional): best-bets, technical-indicators, calendar-spreads, signal-dashboard, qa-dashboard, research
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, FeatureNode] = {}
        self._dependencies: List[Dependency] = []
        self._initialize_graph()

    def _initialize_graph(self) -> None:
        """Initialize the dependency graph with known feature relationships"""

        # Define all features by tier
        tier1_features = ['positions', 'dashboard', 'premium-scanner', 'options-analysis', 'game-cards']
        tier2_features = ['ava-chatbot', 'kalshi-markets', 'earnings-calendar', 'xtrades-watchlists', 'dte-scanner']
        tier3_features = ['best-bets', 'technical-indicators', 'calendar-spreads', 'signal-dashboard', 'qa-dashboard', 'research']

        # Create nodes
        for feature in tier1_features:
            self._nodes[feature] = FeatureNode(name=feature, tier=1)
        for feature in tier2_features:
            self._nodes[feature] = FeatureNode(name=feature, tier=2)
        for feature in tier3_features:
            self._nodes[feature] = FeatureNode(name=feature, tier=3)

        # Define dependencies
        self._define_dependencies()

    def _define_dependencies(self) -> None:
        """Define all known feature dependencies"""

        # Dashboard depends on positions for portfolio data
        self._add_dependency(
            from_feature='dashboard',
            to_feature='positions',
            dep_type=DependencyType.DATA_FROM,
            description='Dashboard displays position summary from positions feature',
            critical=True,
        )

        # Dashboard uses premium scanner data
        self._add_dependency(
            from_feature='dashboard',
            to_feature='premium-scanner',
            dep_type=DependencyType.DATA_FROM,
            description='Dashboard shows premium opportunities',
            critical=False,
        )

        # Premium scanner depends on positions for held symbols
        self._add_dependency(
            from_feature='premium-scanner',
            to_feature='positions',
            dep_type=DependencyType.DATA_FROM,
            description='Premium scanner filters by held positions',
            critical=False,
        )

        # Premium scanner uses earnings calendar for avoidance
        self._add_dependency(
            from_feature='premium-scanner',
            to_feature='earnings-calendar',
            dep_type=DependencyType.DATA_FROM,
            description='Earnings avoidance in premium recommendations',
            critical=False,
        )

        # Options analysis requires positions
        self._add_dependency(
            from_feature='options-analysis',
            to_feature='positions',
            dep_type=DependencyType.DATA_FROM,
            description='Options analysis based on current positions',
            critical=True,
        )

        # DTE scanner uses earnings calendar
        self._add_dependency(
            from_feature='dte-scanner',
            to_feature='earnings-calendar',
            dep_type=DependencyType.DATA_FROM,
            description='DTE scanner avoids earnings dates',
            critical=False,
        )

        # Calendar spreads uses options analysis
        self._add_dependency(
            from_feature='calendar-spreads',
            to_feature='options-analysis',
            dep_type=DependencyType.SHARES_API,
            description='Calendar spreads share options chain API',
            critical=True,
        )

        # Best bets uses game cards data
        self._add_dependency(
            from_feature='best-bets',
            to_feature='game-cards',
            dep_type=DependencyType.DATA_FROM,
            description='Best bets derives from game analysis',
            critical=True,
        )

        # Best bets uses Kalshi markets
        self._add_dependency(
            from_feature='best-bets',
            to_feature='kalshi-markets',
            dep_type=DependencyType.DATA_FROM,
            description='Best bets includes Kalshi opportunities',
            critical=False,
        )

        # Signal dashboard aggregates from multiple sources
        self._add_dependency(
            from_feature='signal-dashboard',
            to_feature='technical-indicators',
            dep_type=DependencyType.DATA_FROM,
            description='Signals based on technical indicators',
            critical=True,
        )

        # Technical indicators uses watchlist symbols
        self._add_dependency(
            from_feature='technical-indicators',
            to_feature='xtrades-watchlists',
            dep_type=DependencyType.DATA_FROM,
            description='Technical analysis on watchlist symbols',
            critical=False,
        )

        # AVA chatbot can query most features
        for feature in ['positions', 'dashboard', 'premium-scanner', 'earnings-calendar', 'kalshi-markets']:
            self._add_dependency(
                from_feature='ava-chatbot',
                to_feature=feature,
                dep_type=DependencyType.DATA_FROM,
                description=f'AVA can query {feature} data',
                critical=False,
            )

        # QA dashboard monitors all features
        for feature in self._nodes.keys():
            if feature != 'qa-dashboard':
                self._add_dependency(
                    from_feature='qa-dashboard',
                    to_feature=feature,
                    dep_type=DependencyType.DATA_FROM,
                    description=f'QA monitors {feature} health',
                    critical=False,
                )

        # Research uses RAG across documentation
        self._add_dependency(
            from_feature='research',
            to_feature='positions',
            dep_type=DependencyType.SHARES_DB,
            description='Research accesses position context for queries',
            critical=False,
        )

    def _add_dependency(
        self,
        from_feature: str,
        to_feature: str,
        dep_type: DependencyType,
        description: str = "",
        critical: bool = False,
    ):
        """Add a dependency to the graph"""
        dep = Dependency(
            from_feature=from_feature,
            to_feature=to_feature,
            dependency_type=dep_type,
            description=description,
            critical=critical,
        )
        self._dependencies.append(dep)

        # Update nodes
        if from_feature in self._nodes:
            self._nodes[from_feature].dependencies.append(dep)
        if to_feature in self._nodes:
            self._nodes[to_feature].dependents.append(from_feature)

    def get_dependencies(self, feature_name: str) -> List[Dependency]:
        """Get all dependencies for a feature"""
        if feature_name not in self._nodes:
            return []
        return self._nodes[feature_name].dependencies

    def get_dependents(self, feature_name: str) -> List[str]:
        """Get features that depend on this feature"""
        if feature_name not in self._nodes:
            return []
        return self._nodes[feature_name].dependents

    def get_critical_dependencies(self, feature_name: str) -> List[str]:
        """Get critical dependencies (blockers) for a feature"""
        deps = self.get_dependencies(feature_name)
        return [d.to_feature for d in deps if d.critical]

    def get_execution_order(self) -> List[str]:
        """
        Get optimal test execution order based on dependencies.
        Features with no dependencies run first.
        """
        # Topological sort
        in_degree = {name: 0 for name in self._nodes}

        for dep in self._dependencies:
            if dep.from_feature in in_degree:
                in_degree[dep.from_feature] += 1

        # Start with features that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        queue.sort(key=lambda x: self._nodes[x].tier)  # Sort by tier

        result = []
        while queue:
            # Pick lowest tier feature
            queue.sort(key=lambda x: (self._nodes[x].tier, x))
            feature = queue.pop(0)
            result.append(feature)

            # Reduce in-degree for dependents
            for dep in self._dependencies:
                if dep.to_feature == feature and dep.from_feature in in_degree:
                    in_degree[dep.from_feature] -= 1
                    if in_degree[dep.from_feature] == 0:
                        queue.append(dep.from_feature)

        # Add any remaining features (circular dependencies)
        for feature in self._nodes:
            if feature not in result:
                result.append(feature)

        return result

    def analyze_cascade_impact(self, failed_feature: str) -> Dict[str, Any]:
        """
        Analyze the cascade impact of a feature failure.

        Returns:
            Dict with affected features and severity assessment
        """
        affected = set()
        critical_affected = set()

        # BFS to find all affected features
        queue = [failed_feature]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for dependent in self.get_dependents(current):
                affected.add(dependent)

                # Check if this is a critical dependency
                deps = self.get_dependencies(dependent)
                for dep in deps:
                    if dep.to_feature == current and dep.critical:
                        critical_affected.add(dependent)
                        queue.append(dependent)

        # Calculate severity
        tier1_affected = [f for f in affected if self._nodes.get(f, FeatureNode('', 3)).tier == 1]
        tier2_affected = [f for f in affected if self._nodes.get(f, FeatureNode('', 3)).tier == 2]

        if tier1_affected:
            severity = 'critical'
        elif critical_affected:
            severity = 'high'
        elif tier2_affected:
            severity = 'medium'
        else:
            severity = 'low'

        return {
            'failed_feature': failed_feature,
            'total_affected': len(affected),
            'affected_features': list(affected),
            'critical_affected': list(critical_affected),
            'tier1_affected': tier1_affected,
            'tier2_affected': tier2_affected,
            'severity': severity,
            'message': f"Failure in {failed_feature} affects {len(affected)} features ({len(critical_affected)} critically)",
        }

    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a feature's dependencies"""
        if feature_name not in self._nodes:
            return None

        node = self._nodes[feature_name]
        deps = self.get_dependencies(feature_name)
        dependents = self.get_dependents(feature_name)

        return {
            'name': feature_name,
            'tier': node.tier,
            'dependency_count': len(deps),
            'dependent_count': len(dependents),
            'dependencies': [d.to_dict() for d in deps],
            'dependents': dependents,
            'critical_dependencies': self.get_critical_dependencies(feature_name),
        }

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary of the entire dependency graph"""
        return {
            'total_features': len(self._nodes),
            'total_dependencies': len(self._dependencies),
            'tier1_features': [n for n, node in self._nodes.items() if node.tier == 1],
            'tier2_features': [n for n, node in self._nodes.items() if node.tier == 2],
            'tier3_features': [n for n, node in self._nodes.items() if node.tier == 3],
            'execution_order': self.get_execution_order(),
            'critical_dependencies': sum(1 for d in self._dependencies if d.critical),
        }

    def visualize_ascii(self) -> str:
        """Generate ASCII visualization of the dependency graph"""
        lines = []
        lines.append("=" * 60)
        lines.append("Feature Dependency Graph")
        lines.append("=" * 60)

        for tier in [1, 2, 3]:
            lines.append(f"\n--- Tier {tier} ---")
            features = [n for n, node in self._nodes.items() if node.tier == tier]

            for feature in sorted(features):
                deps = self.get_dependencies(feature)
                dependents = self.get_dependents(feature)

                lines.append(f"\n[{feature}]")
                if deps:
                    lines.append("  Depends on:")
                    for dep in deps:
                        critical = " (CRITICAL)" if dep.critical else ""
                        lines.append(f"    -> {dep.to_feature}{critical}")
                if dependents:
                    lines.append("  Dependents:")
                    for dep in dependents:
                        lines.append(f"    <- {dep}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# Singleton instance
_dependency_graph: Optional[FeatureDependencyGraph] = None


def get_dependency_graph() -> FeatureDependencyGraph:
    """Get the singleton dependency graph instance"""
    global _dependency_graph
    if _dependency_graph is None:
        _dependency_graph = FeatureDependencyGraph()
    return _dependency_graph
