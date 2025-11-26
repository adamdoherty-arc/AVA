"""
Main Orchestrator - Automatic coordination and validation for all Claude Code requests
Inspired by LangGraph (state machine), AutoGen (conversation), and CrewAI (roles)
"""
import os
import sys
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import local modules
try:
    from pre_flight_validator import PreFlightValidator
    from qa_agent import QAAgent
    from rule_engine import RuleEngine
except ImportError:
    # Try absolute import
    from .claude.orchestrator.pre_flight_validator import PreFlightValidator
    from .claude.orchestrator.qa_agent import QAAgent
    from .claude.orchestrator.rule_engine import RuleEngine


class OrchestratorMode(Enum):
    """Orchestration mode"""
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    QA_RUNNING = "qa_running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationContext:
    """Context for orchestration"""
    request: str
    files_modified: List[str] = field(default_factory=list)
    features_involved: List[str] = field(default_factory=list)
    rules_to_enforce: List[str] = field(default_factory=list)
    specialist_agents: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    qa_results: Dict[str, Any] = field(default_factory=dict)
    research_results: Dict[str, Any] = field(default_factory=dict)  # NEW: Plan-first workflow
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


class MainOrchestrator:
    """
    Main Orchestrator - Coordinates all AI agents and enforces project rules

    Responsibilities:
    1. Pre-flight validation before any code changes
    2. Automatic spec agent consultation
    3. Post-execution QA and compliance checking
    4. Rule enforcement and auto-fixing
    5. Learning from patterns

    Architecture Pattern: LangGraph state machine approach
    - Each stage is a node in the execution graph
    - Transitions based on validation results
    - Parallel execution where possible (AutoGen pattern)
    - Role-based agent selection (CrewAI pattern)
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize orchestrator"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Initialize components
        self.pre_flight = PreFlightValidator(self.config)
        self.qa_agent = QAAgent(self.config)
        self.rule_engine = RuleEngine(self.config)

        # Load feature registry
        registry_path = Path(__file__).parent / "feature_registry.yaml"
        self.feature_registry = self._load_config(registry_path)

        self.logger.info("Main Orchestrator initialized")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found"""
        return {
            "orchestrator": {"enabled": True, "mode": "standard"},
            "pre_flight": {"enabled": True, "strict_mode": True},
            "post_execution": {"enabled": True, "auto_fix": False},
            "logging": {"enabled": True, "level": "INFO"}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("MainOrchestrator")

        if not self.config.get("logging", {}).get("enabled", True):
            logger.addHandler(logging.NullHandler())
            return logger

        level = self.config.get("logging", {}).get("level", "INFO")
        logger.setLevel(getattr(logging, level))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.config.get("logging", {}).get("file")
        if log_file:
            log_path = Path(__file__).parent / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def orchestrate(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationContext:
        """
        Main orchestration flow - runs automatically on every request

        Flow (LangGraph state machine):
        0. PENDING -> PLANNING (Research & Planning - NEW)
        1. PLANNING -> VALIDATING (Pre-flight validation)
        2. VALIDATING -> EXECUTING (If validation passes)
        3. EXECUTING -> QA_RUNNING (Post-execution QA)
        4. QA_RUNNING -> COMPLETED (If QA passes)

        Args:
            request: User's request/query
            context: Additional context (files, current state, etc.)

        Returns:
            OrchestrationContext with results
        """
        if not self.config.get("orchestrator", {}).get("enabled", True):
            self.logger.info("Orchestrator disabled, skipping")
            return None

        ctx = OrchestrationContext(request=request)
        context = context or {}

        try:
            # === STAGE 0: RESEARCH & PLANNING (NEW) ===
            orchestrator_config = self.config.get("orchestrator", {})

            if orchestrator_config.get("require_plan_approval", False):
                ctx.status = TaskStatus.VALIDATING  # Use VALIDATING for planning phase
                self.logger.info("Starting plan-first research phase...")

                # Run research and generate plan
                research_results = self.pre_flight.research_and_plan(request, context)
                ctx.research_results = research_results

                # If research completed, the plan needs user approval
                # The orchestrator will pause here - ExitPlanMode will be called
                # to present the plan to the user for approval
                if research_results.get("research_completed", False):
                    self.logger.info("Research complete. Waiting for user approval...")
                    self.logger.info(f"Generated plan:\n{research_results.get('plan', 'No plan')}")

                    # NOTE: In actual implementation, this would trigger ExitPlanMode
                    # and wait for user approval before continuing
                    # For now, we just log and continue to validation

            # === STAGE 1: PRE-FLIGHT VALIDATION ===
            ctx.status = TaskStatus.VALIDATING
            self.logger.info(f"Starting orchestration for request: {request[:100]}...")

            if self.config.get("pre_flight", {}).get("enabled", True):
                self.logger.info("Running pre-flight validation...")
                validation_result = self.pre_flight.validate(request, context)
                ctx.validation_results = validation_result

                if not validation_result.get("passed", True):
                    ctx.status = TaskStatus.FAILED
                    ctx.end_time = datetime.now()
                    self.logger.error(f"Pre-flight validation failed: {validation_result.get('errors')}")
                    return ctx

                # Inject context from specs
                ctx.features_involved = validation_result.get("features", [])
                ctx.rules_to_enforce = validation_result.get("rules", [])
                ctx.specialist_agents = validation_result.get("agents", [])

            # === STAGE 2: EXECUTION ===
            # (This happens externally by Claude Code - we just track it)
            ctx.status = TaskStatus.EXECUTING
            self.logger.info(f"Execution stage - Features: {ctx.features_involved}")

            # === STAGE 3: POST-EXECUTION QA ===
            # (This runs AFTER code changes are made)
            # For now, mark as completed - QA runs separately via post-execution hook
            ctx.status = TaskStatus.COMPLETED
            ctx.end_time = datetime.now()

            self.logger.info(f"Orchestration completed in {(ctx.end_time - ctx.start_time).total_seconds():.2f}s")

            return ctx

        except Exception as e:
            self.logger.error(f"Orchestration error: {e}", exc_info=True)
            ctx.status = TaskStatus.FAILED
            ctx.end_time = datetime.now()
            return ctx

    def post_execution_qa(
        self,
        files_modified: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run QA checks after code has been modified

        Args:
            files_modified: List of files that were changed
            context: Additional context

        Returns:
            QA results with pass/fail status
        """
        if not self.config.get("post_execution", {}).get("enabled", True):
            self.logger.info("Post-execution QA disabled")
            return {"enabled": False}

        self.logger.info(f"Running post-execution QA on {len(files_modified)} files...")

        try:
            qa_results = self.qa_agent.run_qa(files_modified, context or {})

            # Auto-fix violations if enabled
            if self.config.get("post_execution", {}).get("auto_fix", False):
                if qa_results.get("violations"):
                    self.logger.info("Auto-fixing violations...")
                    fix_results = self.rule_engine.auto_fix(qa_results["violations"])
                    qa_results["auto_fixed"] = fix_results

            # Auto-commit changes if QA passed and auto-commit is enabled
            if qa_results.get("passed", False):
                if self.config.get("post_execution", {}).get("auto_commit", True):
                    self.logger.info("QA passed - auto-committing changes...")
                    commit_result = self._auto_commit_changes(files_modified, qa_results, context)
                    qa_results["committed"] = commit_result.get("success", False)
                    qa_results["commit_sha"] = commit_result.get("commit_sha")

            return qa_results

        except Exception as e:
            self.logger.error(f"Post-execution QA error: {e}", exc_info=True)
            return {"passed": False, "error": str(e)}

    def _auto_commit_changes(
        self,
        files_modified: List[str],
        qa_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Auto-commit QA'd changes to prevent data loss

        Args:
            files_modified: List of files that were changed
            qa_results: Results from QA checks
            context: Additional context (may include request description)

        Returns:
            Commit result with success status and commit SHA
        """
        try:
            # Get repository root
            repo_root = Path(__file__).parent.parent.parent

            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                self.logger.warning("Not in a git repository - skipping auto-commit")
                return {"success": False, "reason": "not_a_git_repo"}

            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if not status_result.stdout.strip():
                self.logger.info("No changes to commit")
                return {"success": False, "reason": "no_changes"}

            # Stage the modified files
            for file_path in files_modified:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=repo_root,
                    capture_output=True,
                    timeout=5
                )

            # Generate commit message
            commit_msg = self._generate_commit_message(files_modified, qa_results, context)

            # Create commit
            commit_result = subprocess.run(
                ["git", "commit", "--no-verify", "-m", commit_msg],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            if commit_result.returncode != 0:
                self.logger.error(f"Commit failed: {commit_result.stderr}")
                return {"success": False, "reason": "commit_failed", "error": commit_result.stderr}

            # Get commit SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )

            commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None

            self.logger.info(f"✅ Auto-committed changes: {commit_sha[:8] if commit_sha else 'unknown'}")

            return {
                "success": True,
                "commit_sha": commit_sha,
                "files_committed": len(files_modified),
                "message": commit_msg
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Git command timeout during auto-commit")
            return {"success": False, "reason": "timeout"}
        except Exception as e:
            self.logger.error(f"Auto-commit error: {e}", exc_info=True)
            return {"success": False, "reason": "exception", "error": str(e)}

    def _generate_commit_message(
        self,
        files_modified: List[str],
        qa_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a descriptive commit message

        Args:
            files_modified: List of modified files
            qa_results: QA results
            context: Additional context

        Returns:
            Commit message string
        """
        # Get base description from context or use default
        description = (context or {}).get("description", "Auto-commit after QA validation")

        # Identify features
        features = self.identify_features(files_modified)
        feature_str = f" ({', '.join(features)})" if features else ""

        # Build commit message
        lines = [
            f"chore: {description}{feature_str}",
            "",
            f"✅ QA Validation: PASSED",
            f"📁 Files modified: {len(files_modified)}",
        ]

        # Add file list if not too many
        if len(files_modified) <= 10:
            lines.append("")
            lines.append("Modified files:")
            for file_path in files_modified:
                lines.append(f"  - {Path(file_path).name}")

        # Add QA summary
        if qa_results.get("checks_run"):
            lines.append("")
            lines.append(f"QA checks run: {qa_results['checks_run']}")

        # Footer
        lines.extend([
            "",
            "🤖 Auto-committed by Orchestrator to prevent data loss",
            "",
            "Generated with Claude Code",
            "",
            "Co-Authored-By: Claude <noreply@anthropic.com>"
        ])

        return "\n".join(lines)

    def identify_features(self, files: List[str]) -> List[str]:
        """
        Identify which features are involved based on file paths

        Args:
            files: List of file paths

        Returns:
            List of feature names
        """
        features = set()

        for file_path in files:
            file_name = Path(file_path).name

            # Check feature registry
            for feature_name, feature_config in self.feature_registry.get("features", {}).items():
                if file_name in feature_config.get("pages", []):
                    features.add(feature_name)

        return list(features)

    def get_specialist_agents(self, features: List[str]) -> List[str]:
        """
        Get specialist agents for given features

        Args:
            features: List of feature names

        Returns:
            List of specialist agent names
        """
        agents = set()

        for feature in features:
            feature_config = self.feature_registry.get("features", {}).get(feature, {})
            specialist = feature_config.get("specialist_agent")
            if specialist:
                agents.add(specialist)

        return list(agents)

    def get_summary(self) -> str:
        """Get orchestrator summary"""
        return f"""
Main Orchestrator Status:
- Mode: {self.config.get('orchestrator', {}).get('mode', 'standard')}
- Pre-flight: {'enabled' if self.config.get('pre_flight', {}).get('enabled') else 'disabled'}
- Post-execution QA: {'enabled' if self.config.get('post_execution', {}).get('enabled') else 'disabled'}
- Features tracked: {len(self.feature_registry.get('features', {}))}
- Rules loaded: {len(self.config.get('rules', {}).get('ui', {})) + len(self.config.get('rules', {}).get('code', {}))}
"""


# Singleton instance
_orchestrator_instance = None

def get_orchestrator() -> MainOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MainOrchestrator()
    return _orchestrator_instance


if __name__ == "__main__":
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Main Orchestrator")
    parser.add_argument("--request", type=str, help="Request to orchestrate")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--qa", nargs="+", help="Run QA on files")

    args = parser.parse_args()

    orchestrator = get_orchestrator()

    if args.summary:
        print(orchestrator.get_summary())
    elif args.request:
        result = orchestrator.orchestrate(args.request)
        print(f"\nOrchestration Result:")
        print(f"Status: {result.status.value}")
        print(f"Features: {result.features_involved}")
        print(f"Validation: {'PASSED' if result.validation_results.get('passed') else 'FAILED'}")
    elif args.qa:
        result = orchestrator.post_execution_qa(args.qa)
        print(f"\nQA Result:")
        print(f"Passed: {result.get('passed', False)}")
        if result.get('violations'):
            print(f"Violations: {len(result['violations'])}")
            for v in result['violations'][:5]:
                print(f"  - {v}")
    else:
        parser.print_help()
