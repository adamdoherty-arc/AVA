"""
Self-Healing Engine for Orchestrator
Auto-fixes common errors and retries with different approaches
Uses local LLM (Ollama) for error analysis
"""
from typing import Dict, Any, List, Optional
import logging
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SelfHealer:
    """
    Self-healing capabilities for orchestrator
    - Analyzes failures
    - Applies known fixes
    - Retries with different strategies
    """

    def __init__(self) -> None:
        self.fix_patterns = self._load_fix_patterns()
        self.max_retries = 3
        logger.info("Self-healer initialized")

    def _load_fix_patterns(self) -> Dict[str, Any]:
        """Load known fix patterns"""
        return {
            "import_error": {
                "pattern": r"No module named '(\w+)'",
                "fix_type": "install_package",
                "confidence": 0.9
            },
            "syntax_error": {
                "pattern": r"SyntaxError:",
                "fix_type": "syntax_fix",
                "confidence": 0.7
            },
            "indentation_error": {
                "pattern": r"IndentationError:",
                "fix_type": "fix_indentation",
                "confidence": 0.8
            },
            "type_error": {
                "pattern": r"TypeError:",
                "fix_type": "type_fix",
                "confidence": 0.6
            },
            "attribute_error": {
                "pattern": r"AttributeError:",
                "fix_type": "attribute_fix",
                "confidence": 0.5
            },
            "file_not_found": {
                "pattern": r"FileNotFoundError:",
                "fix_type": "create_file",
                "confidence": 0.7
            }
        }

    def can_auto_heal(self, error_message: str, error_type: str) -> bool:
        """Check if error can be auto-healed"""
        for pattern_name, pattern_info in self.fix_patterns.items():
            if re.search(pattern_info["pattern"], error_message):
                confidence = pattern_info["confidence"]
                return confidence >= 0.6  # Only auto-heal high-confidence fixes
        return False

    def suggest_fix(self, error_message: str, error_type: str,
                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest a fix for the error"""
        for pattern_name, pattern_info in self.fix_patterns.items():
            match = re.search(pattern_info["pattern"], error_message)
            if match:
                return {
                    "fix_type": pattern_info["fix_type"],
                    "pattern_name": pattern_name,
                    "confidence": pattern_info["confidence"],
                    "details": match.groups(),
                    "suggestion": self._generate_suggestion(
                        pattern_info["fix_type"],
                        match,
                        context
                    )
                }
        return None

    def _generate_suggestion(self, fix_type: str, match: re.Match,
                            context: Dict[str, Any]) -> str:
        """Generate human-readable fix suggestion"""
        if fix_type == "install_package":
            package = match.group(1) if match.groups() else "unknown"
            return f"Install missing package: pip install {package}"

        elif fix_type == "syntax_fix":
            return "Fix syntax error in the code"

        elif fix_type == "fix_indentation":
            return "Fix indentation (use consistent spaces/tabs)"

        elif fix_type == "type_fix":
            return "Fix type mismatch in function call or assignment"

        elif fix_type == "attribute_fix":
            return "Check object attributes and method names"

        elif fix_type == "create_file":
            return "Create missing file or check file path"

        return "Manual review required"

    def apply_auto_fix(self, error_message: str, error_type: str,
                      context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply automatic fix (for high-confidence patterns only)
        Returns fix details if applied, None otherwise
        """
        fix = self.suggest_fix(error_message, error_type, context)

        if not fix or fix["confidence"] < 0.8:
            return None

        fix_type = fix["fix_type"]
        applied = False

        if fix_type == "install_package":
            # Log suggestion but don't auto-install
            # (security: user should approve package installations)
            logger.info(f"Auto-fix suggestion: {fix['suggestion']}")
            applied = False

        elif fix_type == "syntax_fix":
            # Would require code rewriting - skip for now
            applied = False

        # For now, we'll log suggestions but not apply them automatically
        # This prevents potentially destructive changes

        if applied:
            return {
                "applied": True,
                "fix_type": fix_type,
                "details": fix
            }
        else:
            return {
                "applied": False,
                "suggestion": fix["suggestion"],
                "requires_manual_review": True
            }

    def suggest_retry_strategy(self, failure_context: Dict[str, Any],
                               retry_count: int) -> Dict[str, Any]:
        """
        Suggest retry strategy based on failure context
        """
        agent_name = failure_context.get("agent_name")
        error_type = failure_context.get("error_type")

        strategies = {
            "different_agent": {
                "description": "Try a different agent for the same task",
                "confidence": 0.7
            },
            "simpler_approach": {
                "description": "Break task into smaller subtasks",
                "confidence": 0.8
            },
            "different_model": {
                "description": "Use a different local model",
                "confidence": 0.6
            },
            "add_context": {
                "description": "Provide more context to the agent",
                "confidence": 0.7
            }
        }

        # Select strategy based on retry count
        if retry_count == 0:
            return strategies["different_agent"]
        elif retry_count == 1:
            return strategies["simpler_approach"]
        else:
            return strategies["add_context"]

    def analyze_failure_with_llm(self, error_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use local LLM (Ollama) to analyze complex failures
        NOTE: Requires Ollama to be running locally
        """
        try:
            # This would integrate with Ollama for complex analysis
            # For now, return structured analysis
            return {
                "analysis": "Complex error requiring manual review",
                "suggested_actions": [
                    "Review error message and context",
                    "Check recent code changes",
                    "Verify dependencies and configuration"
                ],
                "confidence": 0.5
            }
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "analysis": "LLM analysis unavailable",
                "error": str(e)
            }


# Singleton
_healer_instance: Optional[SelfHealer] = None


def get_self_healer() -> SelfHealer:
    """Get singleton self-healer"""
    global _healer_instance
    if _healer_instance is None:
        _healer_instance = SelfHealer()
    return _healer_instance


if __name__ == "__main__":
    # Test self-healer
    healer = get_self_healer()

    # Test import error
    test_error = "No module named 'nonexistent_package'"
    can_heal = healer.can_auto_heal(test_error, "ImportError")
    print(f"\nCan auto-heal import error: {can_heal}")

    fix = healer.suggest_fix(test_error, "ImportError", {})
    print(f"Fix suggestion: {json.dumps(fix, indent=2)}")

    # Test retry strategy
    failure_context = {
        "agent_name": "test-agent",
        "error_type": "ImportError"
    }
    strategy = healer.suggest_retry_strategy(failure_context, retry_count=0)
    print(f"\nRetry strategy: {json.dumps(strategy, indent=2)}")

    print("\nSelf-healer test complete!")
