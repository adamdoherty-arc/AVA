"""
Pre-Flight Validator - Validates requests before execution
Checks against project rules, specs, and known anti-patterns
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class PreFlightValidator:
    """
    Pre-flight validation before code execution

    Checks:
    1. UI style guide compliance
    2. Known anti-patterns
    3. Feature spec existence
    4. Breaking change detection
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator"""
        self.config = config
        self.logger = logging.getLogger("PreFlightValidator")
        self.project_root = Path(__file__).parent.parent.parent

        # Load rules
        self.ui_rules = config.get("rules", {}).get("ui", {})
        self.code_rules = config.get("rules", {}).get("code", {})

    def validate(
        self,
        request: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run pre-flight validation

        Args:
            request: User's request
            context: Current context (files, state, etc.)

        Returns:
            Validation results with passed/failed status
        """
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "features": [],
            "rules": [],
            "agents": [],
            "context_injected": {}
        }

        try:
            # Check 1: Known anti-patterns in request
            self._check_anti_patterns(request, results)

            # Check 2: If modifying files, check against rules
            if context.get("files"):
                self._check_file_rules(context["files"], results)

            # Check 3: Identify features involved
            if context.get("files"):
                results["features"] = self._identify_features(context["files"])

            # Check 4: Load relevant specs
            if results["features"]:
                specs = self._load_feature_specs(results["features"])
                results["context_injected"]["specs"] = specs

            # Check 5: Identify required specialist agents
            if results["features"]:
                results["agents"] = self._get_specialist_agents(results["features"])

            # Check 6: Validate against feature-specific rules
            if results["features"]:
                self._check_feature_rules(results["features"], results)

            # Set overall pass/fail
            results["passed"] = len(results["errors"]) == 0

            if not results["passed"]:
                self.logger.warning(f"Pre-flight validation failed with {len(results['errors'])} errors")
            else:
                self.logger.info(f"Pre-flight validation passed")

            return results

        except Exception as e:
            self.logger.error(f"Pre-flight validation error: {e}", exc_info=True)
            results["passed"] = False
            results["errors"].append(f"Validation error: {str(e)}")
            return results

    def _check_anti_patterns(self, request: str, results: Dict[str, Any]):
        """Check for known anti-patterns in request"""
        request_lower = request.lower()

        # Anti-pattern: Horizontal lines
        forbidden_phrases = ["add divider", "add separator", "add ---", "horizontal line", "horizontal divider"]
        if any(phrase in request_lower for phrase in forbidden_phrases):
            results["errors"].append(
                "FORBIDDEN: Request contains horizontal line/divider. "
                "See UI_STYLE_GUIDE.md - NO horizontal lines allowed."
            )
            results["rules"].append("no_horizontal_lines")

        # Anti-pattern: Hardcoded deltas
        if "delta" in request_lower and any(word in request_lower for word in ["estimate", "hardcode", "assume"]):
            results["warnings"].append(
                "WARNING: Request mentions delta estimation. Use real Greeks from API instead."
            )
            results["rules"].append("use_real_greeks")

        # Anti-pattern: Excel/CSV export
        export_phrases = ["export to excel", "export to csv", "download excel", "download csv",
                         "xlsxwriter", "to_excel", "to_csv", "export data", "download data"]
        if any(phrase in request_lower for phrase in export_phrases):
            results["errors"].append(
                "FORBIDDEN: Excel/CSV export not allowed. Users can copy data directly from tables "
                "or use database exports. See no_data_export rule."
            )
            results["rules"].append("no_data_export")

        # Anti-pattern: Sidebar filters
        sidebar_filter_phrases = ["filters in sidebar", "sidebar filters", "put filters in sidebar",
                                 "move filters to sidebar", "add filter to sidebar"]
        if any(phrase in request_lower for phrase in sidebar_filter_phrases):
            results["errors"].append(
                "FORBIDDEN: Filters must be on main page in expanders or columns, not in sidebar. "
                "See main_page_filters rule."
            )
            results["rules"].append("main_page_filters")

    def _check_file_rules(self, files: List[str], results: Dict[str, Any]):
        """Check files against project rules"""
        for file_path in files:
            if file_path.endswith(".py"):
                # Check if it's a page file
                if "_page" in file_path:
                    results["rules"].append("no_horizontal_lines")
                    results["context_injected"]["ui_style_guide"] = str(
                        self.project_root / "UI_STYLE_GUIDE.md"
                    )

    def _identify_features(self, files: List[str]) -> List[str]:
        """Identify features from file paths"""
        features = set()

        # Load feature registry
        registry_path = self.project_root / ".claude" / "orchestrator" / "feature_registry.yaml"
        if not registry_path.exists():
            return []

        import yaml
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        for file_path in files:
            file_name = Path(file_path).name
            for feature_name, feature_config in registry.get("features", {}).items():
                if file_name in feature_config.get("pages", []):
                    features.add(feature_name)

        return list(features)

    def _load_feature_specs(self, features: List[str]) -> Dict[str, str]:
        """Load specifications for features"""
        specs = {}

        for feature in features:
            spec_path = self.project_root / ".claude" / "specs" / feature
            if spec_path.exists():
                # Look for requirements.md, design.md, etc.
                for spec_file in ["requirements.md", "design.md", "tasks.md"]:
                    file_path = spec_path / spec_file
                    if file_path.exists():
                        specs[f"{feature}/{spec_file}"] = str(file_path)

        return specs

    def _get_specialist_agents(self, features: List[str]) -> List[str]:
        """Get specialist agents for features"""
        agents = set()

        # Load feature registry
        registry_path = self.project_root / ".claude" / "orchestrator" / "feature_registry.yaml"
        if not registry_path.exists():
            return []

        import yaml
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        for feature in features:
            feature_config = registry.get("features", {}).get(feature, {})
            specialist = feature_config.get("specialist_agent")
            if specialist:
                agents.add(specialist)

        return list(agents)

    def _check_feature_rules(self, features: List[str], results: Dict[str, Any]):
        """Check feature-specific rules"""
        # Load feature registry
        registry_path = self.project_root / ".claude" / "orchestrator" / "feature_registry.yaml"
        if not registry_path.exists():
            return

        import yaml
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        for feature in features:
            feature_config = registry.get("features", {}).get(feature, {})
            critical_rules = feature_config.get("critical_rules", [])

            for rule in critical_rules:
                if rule not in results["rules"]:
                    results["rules"].append(rule)

    def get_validation_summary(self, results: Dict[str, Any]) -> str:
        """Get human-readable validation summary"""
        summary = []

        if results["passed"]:
            summary.append("[PASS] Pre-flight validation PASSED")
        else:
            summary.append("[FAIL] Pre-flight validation FAILED")

        if results["errors"]:
            summary.append("\nErrors:")
            for error in results["errors"]:
                summary.append(f"  {error}")

        if results["warnings"]:
            summary.append("\nWarnings:")
            for warning in results["warnings"]:
                summary.append(f"  {warning}")

        if results["features"]:
            summary.append(f"\nFeatures involved: {', '.join(results['features'])}")

        if results["agents"]:
            summary.append(f"Specialist agents: {', '.join(results['agents'])}")

        if results["rules"]:
            summary.append(f"Rules enforced: {', '.join(results['rules'])}")

        return "\n".join(summary)

    # ==================== PLAN-FIRST WORKFLOW (NEW) ====================

    def research_and_plan(
        self,
        request: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Research current implementation and modern patterns before execution

        This is the new plan-first workflow that:
        1. Analyzes current codebase
        2. Researches modern patterns and best practices
        3. Generates detailed plan
        4. Returns plan for user approval

        Args:
            request: User's request
            context: Current context

        Returns:
            Research results with detailed plan
        """
        self.logger.info("Starting plan-first research phase...")

        research_config = self.config.get("orchestrator", {})
        if not research_config.get("plan_research_enabled", False):
            # Research disabled, return empty
            return {
                "research_completed": False,
                "plan": "Research phase disabled. Proceeding with standard execution."
            }

        research_results = {
            "research_completed": True,
            "current_implementation": {},
            "modern_patterns": {},
            "github_examples": {},
            "best_practices": {},
            "plan": "",
            "risks": [],
            "test_strategy": ""
        }

        try:
            # Get research sources and depth
            sources = research_config.get("research_sources", ["current_codebase"])
            depth = research_config.get("research_depth", "medium")

            # Step 1: Analyze current codebase
            if "current_codebase" in sources:
                self.logger.info("Analyzing current codebase...")
                research_results["current_implementation"] = self._analyze_current_code(
                    request, context
                )

            # Step 2: Research GitHub projects (if enabled)
            if "github_projects" in sources:
                self.logger.info("Researching GitHub projects...")
                research_results["github_examples"] = self._research_github_projects(
                    request, depth
                )

            # Step 3: Research modern frameworks (if enabled)
            if "modern_frameworks" in sources:
                self.logger.info("Researching modern frameworks...")
                research_results["modern_patterns"] = self._research_frameworks(
                    request, depth
                )

            # Step 4: Research best practices (if enabled)
            if any(src in sources for src in ["modern_frameworks", "documentation"]):
                self.logger.info("Researching best practices...")
                research_results["best_practices"] = self._research_best_practices(
                    request, depth
                )

            # Step 5: Synthesize into detailed plan
            self.logger.info("Synthesizing research into plan...")
            research_results["plan"] = self._synthesize_plan(
                request, research_results
            )

            # Step 6: Identify risks
            research_results["risks"] = self._identify_risks(
                request, research_results
            )

            # Step 7: Generate test strategy
            research_results["test_strategy"] = self._generate_test_strategy(
                request, research_results
            )

            self.logger.info("Research phase complete!")
            return research_results

        except Exception as e:
            self.logger.error(f"Research phase error: {e}", exc_info=True)
            research_results["research_completed"] = False
            research_results["plan"] = f"Research error: {str(e)}. Proceeding with standard validation."
            return research_results

    def _analyze_current_code(
        self,
        request: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze current implementation in codebase

        Uses Grep/Glob/Read tools to understand:
        - What files are involved
        - Current implementation patterns
        - Existing functionality
        - Code structure
        """
        analysis = {
            "files_found": [],
            "patterns_detected": [],
            "functionality": [],
            "structure": "",
            "notes": []
        }

        try:
            # Extract key terms from request
            keywords = self._extract_keywords(request)

            # Note: Actual Grep/Glob/Read calls would happen in main_orchestrator
            # This method prepares the analysis structure
            analysis["notes"].append(
                f"Need to search for: {', '.join(keywords)}"
            )
            analysis["notes"].append(
                "Orchestrator will use Grep/Glob to find relevant files"
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Code analysis error: {e}")
            analysis["notes"].append(f"Analysis error: {str(e)}")
            return analysis

    def _research_github_projects(
        self,
        request: str,
        depth: str
    ) -> Dict[str, Any]:
        """
        Research similar implementations on GitHub

        Uses WebSearch to find:
        - Popular implementations
        - Common patterns
        - Best practices from real projects
        """
        research = {
            "search_queries": [],
            "examples_found": [],
            "patterns": [],
            "notes": []
        }

        try:
            # Generate search queries based on request
            queries = self._generate_github_queries(request, depth)
            research["search_queries"] = queries

            # Note: Actual WebSearch calls happen in main_orchestrator
            research["notes"].append(
                "Orchestrator will search GitHub for similar implementations"
            )

            return research

        except Exception as e:
            self.logger.error(f"GitHub research error: {e}")
            research["notes"].append(f"Research error: {str(e)}")
            return research

    def _research_frameworks(
        self,
        request: str,
        depth: str
    ) -> Dict[str, Any]:
        """
        Research modern framework patterns and best practices

        Uses WebSearch to find:
        - 2025 best practices
        - Modern framework patterns
        - Architecture recommendations
        """
        research = {
            "search_queries": [],
            "patterns_found": [],
            "recommendations": [],
            "notes": []
        }

        try:
            # Generate framework-specific queries
            queries = self._generate_framework_queries(request, depth)
            research["search_queries"] = queries

            # Note: Actual WebSearch calls happen in main_orchestrator
            research["notes"].append(
                "Orchestrator will research modern framework patterns"
            )

            return research

        except Exception as e:
            self.logger.error(f"Framework research error: {e}")
            research["notes"].append(f"Research error: {str(e)}")
            return research

    def _research_best_practices(
        self,
        request: str,
        depth: str
    ) -> Dict[str, Any]:
        """
        Research best practices for the requested feature

        Uses WebSearch to find:
        - Industry best practices
        - Common pitfalls
        - Performance optimization tips
        """
        research = {
            "search_queries": [],
            "practices": [],
            "pitfalls": [],
            "notes": []
        }

        try:
            # Generate best practice queries
            queries = self._generate_best_practice_queries(request, depth)
            research["search_queries"] = queries

            # Note: Actual WebSearch calls happen in main_orchestrator
            research["notes"].append(
                "Orchestrator will research best practices"
            )

            return research

        except Exception as e:
            self.logger.error(f"Best practice research error: {e}")
            research["notes"].append(f"Research error: {str(e)}")
            return research

    def _synthesize_plan(
        self,
        request: str,
        research_results: Dict[str, Any]
    ) -> str:
        """
        Synthesize research into detailed execution plan

        Combines all research to create:
        - Current state analysis
        - Proposed changes with file paths and line numbers
        - Modern patterns to apply
        - Step-by-step implementation plan
        """
        plan_parts = []

        plan_parts.append("# DETAILED EXECUTION PLAN")
        plan_parts.append(f"\n## Request\n{request}")

        # Current implementation
        if research_results.get("current_implementation"):
            plan_parts.append("\n## Current Implementation")
            impl = research_results["current_implementation"]
            if impl.get("files_found"):
                plan_parts.append(f"Files involved: {', '.join(impl['files_found'])}")
            if impl.get("notes"):
                plan_parts.append("\n".join(impl["notes"]))

        # Modern patterns to apply
        if research_results.get("modern_patterns"):
            plan_parts.append("\n## Modern Patterns to Apply")
            patterns = research_results["modern_patterns"]
            if patterns.get("recommendations"):
                for rec in patterns["recommendations"]:
                    plan_parts.append(f"- {rec}")

        # Best practices
        if research_results.get("best_practices"):
            plan_parts.append("\n## Best Practices")
            practices = research_results["best_practices"]
            if practices.get("practices"):
                for practice in practices["practices"]:
                    plan_parts.append(f"- {practice}")

        plan_parts.append("\n## Implementation Steps")
        plan_parts.append("1. [To be filled by orchestrator based on research]")
        plan_parts.append("2. [Steps will be generated after research completes]")

        return "\n".join(plan_parts)

    def _identify_risks(
        self,
        request: str,
        research_results: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risks in the proposed changes"""
        risks = []

        # Common risks
        if "delete" in request.lower() or "remove" in request.lower():
            risks.append("RISK: Deleting code may break dependencies")

        if "database" in request.lower() or "db" in request.lower():
            risks.append("RISK: Database changes may require migration")

        if "breaking" in request.lower():
            risks.append("RISK: Breaking changes may affect users")

        return risks

    def _generate_test_strategy(
        self,
        request: str,
        research_results: Dict[str, Any]
    ) -> str:
        """Generate testing strategy based on changes"""
        strategy_parts = []

        strategy_parts.append("# TEST STRATEGY")
        strategy_parts.append("\n## Unit Tests")
        strategy_parts.append("- Test new functions individually")

        strategy_parts.append("\n## Integration Tests")
        strategy_parts.append("- Test full feature workflow")

        strategy_parts.append("\n## Manual Testing")
        strategy_parts.append("- Run application and verify UI")
        strategy_parts.append("- Check all edge cases")

        return "\n".join(strategy_parts)

    # Helper methods for query generation

    def _extract_keywords(self, request: str) -> List[str]:
        """Extract key terms from request for searching"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = request.lower().split()
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:10]  # Top 10 keywords

    def _generate_github_queries(self, request: str, depth: str) -> List[str]:
        """Generate GitHub search queries"""
        keywords = self._extract_keywords(request)
        queries = []

        if depth == "quick":
            # 1-2 queries
            queries.append(f"site:github.com {' '.join(keywords[:3])}")
        elif depth == "medium":
            # 3-5 queries
            queries.append(f"site:github.com {' '.join(keywords[:3])} python")
            queries.append(f"site:github.com {' '.join(keywords[:3])} streamlit")
        else:  # deep
            # 5-10 queries
            queries.append(f"site:github.com {' '.join(keywords[:3])} python")
            queries.append(f"site:github.com {' '.join(keywords[:3])} streamlit")
            queries.append(f"site:github.com {' '.join(keywords[:3])} dashboard")

        return queries

    def _generate_framework_queries(self, request: str, depth: str) -> List[str]:
        """Generate framework/pattern search queries"""
        keywords = self._extract_keywords(request)
        queries = []

        if depth == "quick":
            queries.append(f"{' '.join(keywords[:2])} best practices 2025")
        elif depth == "medium":
            queries.append(f"{' '.join(keywords[:2])} best practices 2025")
            queries.append(f"{' '.join(keywords[:2])} modern patterns")
        else:  # deep
            queries.append(f"{' '.join(keywords[:2])} best practices 2025")
            queries.append(f"{' '.join(keywords[:2])} modern patterns")
            queries.append(f"{' '.join(keywords[:2])} architecture design")

        return queries

    def _generate_best_practice_queries(self, request: str, depth: str) -> List[str]:
        """Generate best practice search queries"""
        keywords = self._extract_keywords(request)
        queries = []

        if depth == "quick":
            queries.append(f"{' '.join(keywords[:2])} best practices")
        elif depth == "medium":
            queries.append(f"{' '.join(keywords[:2])} best practices")
            queries.append(f"{' '.join(keywords[:2])} common pitfalls")
        else:  # deep
            queries.append(f"{' '.join(keywords[:2])} best practices")
            queries.append(f"{' '.join(keywords[:2])} common pitfalls")
            queries.append(f"{' '.join(keywords[:2])} performance optimization")

        return queries
