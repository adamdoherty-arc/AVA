import os
"""
SpecAgent Supervisor - Integration with AVA chatbot

Provides natural language interface to SpecAgent system:
- Query health status for specific features
- Run targeted tests on demand
- Report issues found
- Provide feature-specific insights
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from .spec_agent_registry import SpecAgentRegistry
from .spec_orchestrator import SpecAgentOrchestrator
from .base_spec_agent import Issue, IssueSeverity, TestResult

logger = logging.getLogger(__name__)


class SpecAgentState(TypedDict):
    """State for SpecAgent supervisor workflow"""
    messages: List[BaseMessage]
    user_query: str
    query_type: str  # health, test, issues, feature_status
    target_features: List[str]
    test_results: Dict[str, Any]
    issues_found: List[Dict]
    health_scores: Dict[str, float]
    final_response: str


# Tools for AVA integration
@tool
def get_system_health_tool() -> str:
    """Get overall system health from SpecAgents"""
    try:
        supervisor = SpecAgentSupervisor.get_instance()
        if supervisor:
            health = supervisor.get_cached_health()
            return json.dumps(health, indent=2)
        return "SpecAgent system not initialized"
    except Exception as e:
        return f"Error getting health: {str(e)}"


@tool
def get_feature_status_tool(feature_name: str) -> str:
    """Get status of a specific feature from SpecAgents"""
    try:
        supervisor = SpecAgentSupervisor.get_instance()
        if supervisor:
            status = supervisor.get_feature_status(feature_name)
            return json.dumps(status, indent=2)
        return "SpecAgent system not initialized"
    except Exception as e:
        return f"Error getting feature status: {str(e)}"


@tool
def run_feature_test_tool(feature_name: str) -> str:
    """Run tests for a specific feature"""
    try:
        supervisor = SpecAgentSupervisor.get_instance()
        if supervisor:
            # Run async test synchronously
            result = asyncio.get_event_loop().run_until_complete(
                supervisor.run_feature_test(feature_name)
            )
            return json.dumps(result, indent=2)
        return "SpecAgent system not initialized"
    except Exception as e:
        return f"Error running feature test: {str(e)}"


@tool
def list_current_issues_tool() -> str:
    """List all current issues found by SpecAgents"""
    try:
        supervisor = SpecAgentSupervisor.get_instance()
        if supervisor:
            issues = supervisor.get_current_issues()
            return json.dumps(issues, indent=2)
        return "SpecAgent system not initialized"
    except Exception as e:
        return f"Error listing issues: {str(e)}"


class SpecAgentSupervisor:
    """
    Supervisor for SpecAgent system integration with AVA

    Provides:
    - Natural language queries about system health
    - On-demand feature testing
    - Issue reporting and tracking
    - Health score monitoring
    """

    _instance: Optional['SpecAgentSupervisor'] = None

    @classmethod
    def get_instance(cls) -> Optional['SpecAgentSupervisor']:
        """Get singleton instance"""
        return cls._instance

    def __init__(self, base_url: str = os.getenv("API_BASE_URL", "http://localhost:8002").rstrip("/api")):
        """
        Initialize SpecAgent Supervisor

        Args:
            base_url: Base URL for API testing
        """
        self.base_url = base_url
        self.orchestrator = SpecAgentOrchestrator(base_url=base_url)
        self.registry = SpecAgentRegistry()

        # Cache for recent results
        self._cached_results: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        # Current issues
        self._current_issues: List[Dict] = []

        # Build workflow
        self.workflow = self._build_workflow()

        # Set singleton
        SpecAgentSupervisor._instance = self

        logger.info("SpecAgentSupervisor initialized")

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for processing queries"""
        workflow = StateGraph(SpecAgentState)

        # Add nodes
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("get_health", self._get_health_node)
        workflow.add_node("run_tests", self._run_tests_node)
        workflow.add_node("get_issues", self._get_issues_node)
        workflow.add_node("get_feature_status", self._get_feature_status_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Entry point
        workflow.set_entry_point("classify_query")

        # Conditional routing
        workflow.add_conditional_edges(
            "classify_query",
            self._route_query,
            {
                "health": "get_health",
                "test": "run_tests",
                "issues": "get_issues",
                "feature_status": "get_feature_status",
                "direct": "synthesize"
            }
        )

        # All paths lead to synthesize
        workflow.add_edge("get_health", "synthesize")
        workflow.add_edge("run_tests", "synthesize")
        workflow.add_edge("get_issues", "synthesize")
        workflow.add_edge("get_feature_status", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _classify_query_node(self, state: SpecAgentState) -> SpecAgentState:
        """Classify the user query type"""
        query = state["user_query"].lower()

        # Health-related queries
        if any(word in query for word in ["health", "status", "how is", "working"]):
            state["query_type"] = "health"

        # Test-related queries
        elif any(word in query for word in ["test", "check", "verify", "run"]):
            state["query_type"] = "test"

        # Issue-related queries
        elif any(word in query for word in ["issue", "problem", "bug", "error", "wrong"]):
            state["query_type"] = "issues"

        # Feature-specific queries
        elif any(word in query for word in ["dashboard", "positions", "scanner", "options", "sports", "games"]):
            state["query_type"] = "feature_status"
            # Extract feature names
            features = []
            feature_keywords = {
                "dashboard": "dashboard",
                "positions": "positions",
                "scanner": "premium-scanner",
                "premium scanner": "premium-scanner",
                "options": "options-analysis",
                "sports": "game-cards",
                "games": "game-cards",
                "kalshi": "game-cards",
            }
            for keyword, feature in feature_keywords.items():
                if keyword in query:
                    features.append(feature)
            state["target_features"] = features or list(self.registry.get_all().keys())
        else:
            state["query_type"] = "direct"

        logger.info(f"Classified query as: {state['query_type']}")
        return state

    def _route_query(self, state: SpecAgentState) -> str:
        """Route to appropriate handler"""
        return state.get("query_type", "direct")

    def _get_health_node(self, state: SpecAgentState) -> SpecAgentState:
        """Get system health scores"""
        try:
            health = self.get_cached_health()
            state["health_scores"] = health
            state["test_results"] = {"health_check": health}
        except Exception as e:
            logger.error(f"Error getting health: {e}")
            state["health_scores"] = {"error": str(e)}
        return state

    def _run_tests_node(self, state: SpecAgentState) -> SpecAgentState:
        """Run tests for specified features"""
        try:
            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                target_features = state.get("target_features", [])
                if target_features:
                    results = {}
                    for feature in target_features:
                        result = loop.run_until_complete(
                            self.run_feature_test(feature)
                        )
                        results[feature] = result
                else:
                    # Run all priority agents
                    results = loop.run_until_complete(
                        self.orchestrator.run_priority_agents()
                    )
                state["test_results"] = results

                # Extract issues
                issues = []
                for feature, result in results.items():
                    if isinstance(result, dict):
                        for issue in result.get("issues", []):
                            issues.append(issue)
                state["issues_found"] = issues
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            state["test_results"] = {"error": str(e)}
        return state

    def _get_issues_node(self, state: SpecAgentState) -> SpecAgentState:
        """Get current issues"""
        try:
            issues = self.get_current_issues()
            state["issues_found"] = issues
        except Exception as e:
            logger.error(f"Error getting issues: {e}")
            state["issues_found"] = []
        return state

    def _get_feature_status_node(self, state: SpecAgentState) -> SpecAgentState:
        """Get status of specific features"""
        try:
            features = state.get("target_features", [])
            results = {}
            for feature in features:
                status = self.get_feature_status(feature)
                results[feature] = status
            state["test_results"] = results

            # Calculate health scores for these features
            health_scores = {}
            for feature, result in results.items():
                if isinstance(result, dict):
                    health_scores[feature] = result.get("health_score", 0)
            state["health_scores"] = health_scores
        except Exception as e:
            logger.error(f"Error getting feature status: {e}")
            state["test_results"] = {"error": str(e)}
        return state

    def _synthesize_node(self, state: SpecAgentState) -> SpecAgentState:
        """Synthesize results into natural language response"""
        query_type = state.get("query_type", "direct")

        if query_type == "health":
            response = self._format_health_response(state)
        elif query_type == "test":
            response = self._format_test_response(state)
        elif query_type == "issues":
            response = self._format_issues_response(state)
        elif query_type == "feature_status":
            response = self._format_feature_status_response(state)
        else:
            response = "I can help you with system health, feature status, running tests, or viewing current issues. What would you like to know?"

        state["final_response"] = response
        return state

    def _format_health_response(self, state: SpecAgentState) -> str:
        """Format health check response"""
        health = state.get("health_scores", {})

        if "error" in health:
            return f"Unable to get system health: {health['error']}"

        overall = health.get("overall", 0)
        features = health.get("features", {})

        # Build response
        lines = [f"**System Health: {overall:.0f}%**"]

        if overall >= 90:
            lines.append("All systems operating normally.")
        elif overall >= 70:
            lines.append("Minor issues detected.")
        else:
            lines.append("Significant issues require attention.")

        if features:
            lines.append("\n**Feature Health:**")
            for feature, score in sorted(features.items(), key=lambda x: x[1]):
                emoji = "" if score >= 90 else "" if score >= 70 else ""
                lines.append(f"  {emoji} {feature}: {score:.0f}%")

        return "\n".join(lines)

    def _format_test_response(self, state: SpecAgentState) -> str:
        """Format test results response"""
        results = state.get("test_results", {})
        issues = state.get("issues_found", [])

        if "error" in results:
            return f"Test execution failed: {results['error']}"

        lines = ["**Test Results:**"]

        passed = 0
        failed = 0
        for feature, result in results.items():
            if isinstance(result, dict):
                tests_passed = result.get("tests_passed", 0)
                tests_failed = result.get("tests_failed", 0)
                passed += tests_passed
                failed += tests_failed

                if tests_failed == 0:
                    lines.append(f"  {feature}: All {tests_passed} tests passed")
                else:
                    lines.append(f"  {feature}: {tests_passed} passed, {tests_failed} failed")

        if issues:
            lines.append(f"\n**Issues Found ({len(issues)}):**")
            for issue in issues[:5]:  # Limit to 5
                severity = issue.get("severity", "MEDIUM")
                title = issue.get("title", "Unknown issue")
                lines.append(f"  [{severity}] {title}")
            if len(issues) > 5:
                lines.append(f"  ... and {len(issues) - 5} more")

        return "\n".join(lines)

    def _format_issues_response(self, state: SpecAgentState) -> str:
        """Format issues list response"""
        issues = state.get("issues_found", [])

        if not issues:
            return "No issues currently reported by SpecAgents."

        lines = [f"**Current Issues ({len(issues)}):**"]

        # Group by severity
        critical = [i for i in issues if i.get("severity") == "CRITICAL"]
        high = [i for i in issues if i.get("severity") == "HIGH"]
        medium = [i for i in issues if i.get("severity") == "MEDIUM"]
        low = [i for i in issues if i.get("severity") == "LOW"]

        if critical:
            lines.append(f"\n**CRITICAL ({len(critical)}):**")
            for issue in critical:
                lines.append(f"  - {issue.get('title')}: {issue.get('description', '')[:100]}")

        if high:
            lines.append(f"\n**HIGH ({len(high)}):**")
            for issue in high[:3]:
                lines.append(f"  - {issue.get('title')}: {issue.get('description', '')[:100]}")
            if len(high) > 3:
                lines.append(f"  ... and {len(high) - 3} more")

        if medium:
            lines.append(f"\n**MEDIUM ({len(medium)}):**")
            for issue in medium[:3]:
                lines.append(f"  - {issue.get('title')}")
            if len(medium) > 3:
                lines.append(f"  ... and {len(medium) - 3} more")

        if low:
            lines.append(f"\nLOW: {len(low)} issues")

        return "\n".join(lines)

    def _format_feature_status_response(self, state: SpecAgentState) -> str:
        """Format feature status response"""
        results = state.get("test_results", {})
        health = state.get("health_scores", {})

        if not results:
            return "No feature data available."

        lines = ["**Feature Status:**"]

        for feature, result in results.items():
            if isinstance(result, dict):
                score = health.get(feature, 0)
                emoji = "" if score >= 90 else "" if score >= 70 else ""

                lines.append(f"\n**{emoji} {feature}** (Health: {score:.0f}%)")

                # Last test info
                last_run = result.get("last_run", "Unknown")
                lines.append(f"  Last tested: {last_run}")

                # Issues
                feature_issues = result.get("issues", [])
                if feature_issues:
                    lines.append(f"  Issues: {len(feature_issues)}")
                    for issue in feature_issues[:2]:
                        lines.append(f"    - {issue.get('title', 'Unknown')}")
                else:
                    lines.append("  No issues")

        return "\n".join(lines)

    # Public API methods

    def get_cached_health(self) -> Dict[str, Any]:
        """Get cached health scores or refresh if stale"""
        now = datetime.now()

        if (self._cache_timestamp and
            now - self._cache_timestamp < self._cache_ttl and
            "health" in self._cached_results):
            return self._cached_results["health"]

        # Calculate fresh health
        health = {
            "overall": 0,
            "features": {},
            "timestamp": now.isoformat()
        }

        # Get latest results from disk
        data_dir = Path(".claude/orchestrator/continuous_qa/data/spec_agents")
        if data_dir.exists():
            for result_file in data_dir.glob("*.json"):
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        feature = result.get("feature", result_file.stem)
                        score = result.get("health_score", 100)
                        health["features"][feature] = score
                except Exception as e:
                    logger.debug(f"Error reading {result_file}: {e}")

        # Calculate overall
        if health["features"]:
            health["overall"] = sum(health["features"].values()) / len(health["features"])
        else:
            health["overall"] = 100  # No data = assume healthy

        self._cached_results["health"] = health
        self._cache_timestamp = now

        return health

    def get_feature_status(self, feature_name: str) -> Dict[str, Any]:
        """Get status of a specific feature"""
        status = {
            "feature": feature_name,
            "health_score": 100,
            "last_run": None,
            "issues": [],
            "tests_passed": 0,
            "tests_failed": 0
        }

        # Check disk for latest results
        data_dir = Path(".claude/orchestrator/continuous_qa/data/spec_agents")
        result_file = data_dir / f"{feature_name}.json"

        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    status.update(data)
            except Exception as e:
                logger.error(f"Error reading feature status: {e}")

        return status

    async def run_feature_test(self, feature_name: str) -> Dict[str, Any]:
        """Run tests for a specific feature"""
        agent_class = self.registry.get(feature_name)

        if not agent_class:
            return {
                "error": f"Unknown feature: {feature_name}",
                "available_features": list(self.registry.get_all().keys())
            }

        agent = agent_class()
        agent.base_url = self.base_url

        try:
            results = await agent.run_all_tests()

            # Format results
            all_issues = []
            tests_passed = 0
            tests_failed = 0

            for test_result in results:
                if test_result.passed:
                    tests_passed += 1
                else:
                    tests_failed += 1
                for issue in test_result.issues:
                    all_issues.append({
                        "title": issue.title,
                        "description": issue.description,
                        "severity": issue.severity.name,
                        "component": issue.component
                    })

            # Calculate health
            total = tests_passed + tests_failed
            health_score = (tests_passed / total * 100) if total > 0 else 100

            result = {
                "feature": feature_name,
                "health_score": health_score,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "issues": all_issues,
                "last_run": datetime.now().isoformat()
            }

            # Cache result
            self._cache_feature_result(feature_name, result)

            return result

        except Exception as e:
            logger.error(f"Error running tests for {feature_name}: {e}")
            return {
                "feature": feature_name,
                "error": str(e)
            }
        finally:
            await agent.cleanup()

    def _cache_feature_result(self, feature_name: str, result: Dict):
        """Cache feature test result to disk"""
        data_dir = Path(".claude/orchestrator/continuous_qa/data/spec_agents")
        data_dir.mkdir(parents=True, exist_ok=True)

        result_file = data_dir / f"{feature_name}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def get_current_issues(self) -> List[Dict]:
        """Get all current issues from all features"""
        issues = []

        data_dir = Path(".claude/orchestrator/continuous_qa/data/spec_agents")
        if data_dir.exists():
            for result_file in data_dir.glob("*.json"):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        feature_issues = data.get("issues", [])
                        for issue in feature_issues:
                            issue["feature"] = data.get("feature", result_file.stem)
                        issues.extend(feature_issues)
                except Exception as e:
                    logger.debug(f"Error reading {result_file}: {e}")

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        issues.sort(key=lambda x: severity_order.get(x.get("severity", "LOW"), 4))

        return issues

    async def process_query(self, query: str) -> str:
        """
        Process a natural language query about system health

        Args:
            query: User's natural language query

        Returns:
            Natural language response
        """
        initial_state: SpecAgentState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "query_type": "direct",
            "target_features": [],
            "test_results": {},
            "issues_found": [],
            "health_scores": {},
            "final_response": ""
        }

        final_state = await self.workflow.ainvoke(initial_state)
        return final_state.get("final_response", "Unable to process query")

    def process_query_sync(self, query: str) -> str:
        """Synchronous wrapper for process_query"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.process_query(query))


# Export tools for AVA integration
SPEC_AGENT_TOOLS = [
    get_system_health_tool,
    get_feature_status_tool,
    run_feature_test_tool,
    list_current_issues_tool
]
