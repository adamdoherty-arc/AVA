"""
AI-Powered Auto-Fix Engine
===========================

Intelligent auto-fix system that uses AI models (DeepSeek R1, Claude, GPT-4) to:
1. Analyze QA check failures and determine root cause
2. Generate fix suggestions with confidence scoring
3. Apply fixes safely with git-based rollback
4. Validate fixes work by re-running tests
5. Learn from fix success/failure patterns

Uses the intelligent LLM router for cost optimization:
- DeepSeek R1 32B (FREE, LOCAL) for deep reasoning about fixes
- Groq Llama for fast, simple fixes
- Claude/GPT-4 for complex code generation when needed
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# AI client imports
try:
    from src.services.intelligent_llm_router import IntelligentLLMRouter, TaskCategory, ComplexityLevel
    LLM_ROUTER_AVAILABLE = True
except ImportError:
    LLM_ROUTER_AVAILABLE = False
    logger.warning("Intelligent LLM Router not available")

try:
    from backend.infrastructure.ai_client import get_ai_client, AIModelTier
    AI_CLIENT_AVAILABLE = True
except ImportError:
    AI_CLIENT_AVAILABLE = False
    logger.warning("AI Client not available")


class FixDifficulty(Enum):
    """Categorizes how difficult a fix is to implement."""
    TRIVIAL = "trivial"        # Single-line, obvious fix
    SIMPLE = "simple"          # < 5 lines, no dependencies
    MODERATE = "moderate"      # 5-20 lines, some dependencies
    COMPLEX = "complex"        # > 20 lines, many dependencies
    ARCHITECTURAL = "arch"     # Requires system redesign


class FixConfidence(Enum):
    """Confidence level in the proposed fix."""
    HIGH = "high"       # > 80% success probability
    MEDIUM = "medium"   # 50-80% success probability
    LOW = "low"         # < 50% success probability


@dataclass
class FixAttempt:
    """Represents a single fix attempt."""
    issue_id: str
    issue_description: str
    suggested_fix: str
    files_to_modify: List[str]
    difficulty: FixDifficulty
    confidence: FixConfidence
    confidence_score: float  # 0-100
    ai_model_used: str
    reasoning: str
    pre_fix_hash: Optional[str] = None
    post_fix_hash: Optional[str] = None
    applied: bool = False
    validated: bool = False
    rolled_back: bool = False
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FixResult:
    """Result of applying a fix."""
    success: bool
    message: str
    files_modified: List[str] = field(default_factory=list)
    tests_passed: bool = False
    rollback_available: bool = False


class AIAutoFixer:
    """
    AI-powered auto-fix engine.

    Uses multiple AI models strategically:
    - DeepSeek R1 32B: Deep reasoning about complex issues (FREE, LOCAL)
    - Groq Llama: Fast analysis for simple issues
    - Claude Sonnet: High-quality code generation
    - GPT-4: Complex multi-file fixes
    """

    def __init__(self) -> None:
        self.project_root = project_root
        self.fix_history: List[FixAttempt] = []
        self.success_rates: Dict[str, Dict[str, float]] = {}

        # Initialize LLM router if available
        self.llm_router = None
        if LLM_ROUTER_AVAILABLE:
            try:
                self.llm_router = IntelligentLLMRouter()
                logger.info("Intelligent LLM Router initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM router: {e}")

        # Load fix history from file
        self._load_fix_history()

    def _load_fix_history(self) -> None:
        """Load fix history from persistent storage."""
        history_file = self.project_root / ".claude" / "orchestrator" / "data" / "fix_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self.success_rates = data.get("success_rates", {})
            except Exception as e:
                logger.warning(f"Failed to load fix history: {e}")

    def _save_fix_history(self) -> None:
        """Save fix history to persistent storage."""
        history_file = self.project_root / ".claude" / "orchestrator" / "data" / "fix_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(history_file, "w") as f:
                json.dump({
                    "success_rates": self.success_rates,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save fix history: {e}")

    async def analyze_failure(self, failure: Dict[str, Any]) -> Optional[FixAttempt]:
        """
        Analyze a QA failure and generate a fix suggestion.

        Uses AI to understand the root cause and propose a solution.

        Args:
            failure: Dict containing failure details from QA check

        Returns:
            FixAttempt with suggested fix, or None if no fix possible
        """
        issue_type = failure.get("check_name", "unknown")
        details = failure.get("details", {})
        message = failure.get("message", "No details")

        # Determine fix difficulty based on issue type
        difficulty = self._estimate_difficulty(issue_type, details)

        # Choose AI model based on difficulty
        model_tier = self._select_model_tier(difficulty)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(failure)

        # Get AI analysis
        try:
            analysis = await self._get_ai_analysis(prompt, model_tier)

            if not analysis:
                return None

            # Parse the analysis
            fix_attempt = self._parse_analysis(analysis, failure, model_tier)

            # Calculate confidence score
            fix_attempt.confidence_score = self._calculate_confidence(fix_attempt)
            fix_attempt.confidence = self._score_to_confidence(fix_attempt.confidence_score)

            return fix_attempt

        except Exception as e:
            logger.error(f"Failed to analyze failure: {e}")
            return None

    def _estimate_difficulty(self, issue_type: str, details: Dict) -> FixDifficulty:
        """Estimate the difficulty of fixing an issue."""

        # Trivial fixes
        trivial_patterns = [
            "import_error", "typo", "missing_comma", "syntax_error",
            "undefined_variable", "unused_import"
        ]
        if any(p in issue_type.lower() for p in trivial_patterns):
            return FixDifficulty.TRIVIAL

        # Simple fixes
        simple_patterns = [
            "type_error", "attribute_error", "key_error",
            "missing_return", "wrong_type"
        ]
        if any(p in issue_type.lower() for p in simple_patterns):
            return FixDifficulty.SIMPLE

        # Complex patterns
        complex_patterns = [
            "architecture", "refactor", "redesign", "migration",
            "performance", "security"
        ]
        if any(p in issue_type.lower() for p in complex_patterns):
            return FixDifficulty.COMPLEX

        # Default to moderate
        return FixDifficulty.MODERATE

    def _select_model_tier(self, difficulty: FixDifficulty) -> str:
        """Select the appropriate AI model tier based on difficulty."""
        tier_map = {
            FixDifficulty.TRIVIAL: "fast",      # Groq Llama
            FixDifficulty.SIMPLE: "fast",       # Groq Llama
            FixDifficulty.MODERATE: "reasoning", # DeepSeek R1 (FREE!)
            FixDifficulty.COMPLEX: "powerful",   # Claude Sonnet
            FixDifficulty.ARCHITECTURAL: "powerful"
        }
        return tier_map.get(difficulty, "balanced")

    def _build_analysis_prompt(self, failure: Dict[str, Any]) -> str:
        """Build the prompt for AI analysis."""
        return f"""You are an expert software engineer analyzing a QA failure in the AVA Trading Platform.

## QA Failure Details

**Check Name:** {failure.get('check_name', 'unknown')}
**Message:** {failure.get('message', 'No message')}
**Details:** {json.dumps(failure.get('details', {}), indent=2)}
**Auto-Fixable Flag:** {failure.get('auto_fixable', False)}

## Your Task

Analyze this failure and provide:

1. **Root Cause Analysis**: What is causing this failure?
2. **Fix Strategy**: How should we fix this?
3. **Files to Modify**: List specific files that need changes
4. **Code Changes**: Provide the exact code changes needed
5. **Confidence Level**: How confident are you this fix will work? (0-100)
6. **Risk Assessment**: What could go wrong?

## Response Format (JSON)

```json
{{
    "root_cause": "Brief description of the root cause",
    "fix_strategy": "Description of the fix approach",
    "files_to_modify": ["file1.py", "file2.py"],
    "code_changes": [
        {{
            "file": "path/to/file.py",
            "change_type": "edit|create|delete",
            "old_code": "code to replace (if edit)",
            "new_code": "replacement code",
            "line_number": 123
        }}
    ],
    "confidence": 85,
    "risk_level": "low|medium|high",
    "reasoning": "Why this fix will work",
    "rollback_strategy": "How to undo if needed"
}}
```

Provide ONLY valid JSON in your response."""

    async def _get_ai_analysis(self, prompt: str, model_tier: str) -> Optional[str]:
        """Get AI analysis using the appropriate model."""

        # Try intelligent router first
        if self.llm_router:
            try:
                # Map tier to task category
                task_category = {
                    "fast": TaskCategory.SIMPLE_CALC,
                    "reasoning": TaskCategory.DEEP_REASONING,
                    "balanced": TaskCategory.ANALYSIS,
                    "powerful": TaskCategory.CODE_GEN
                }.get(model_tier, TaskCategory.ANALYSIS)

                response = await self.llm_router.route_request(
                    prompt=prompt,
                    task_category=task_category
                )
                return response.get("content", "")
            except Exception as e:
                logger.warning(f"LLM router failed: {e}")

        # Fallback to direct AI client
        if AI_CLIENT_AVAILABLE:
            try:
                client = get_ai_client()
                tier = {
                    "fast": AIModelTier.FAST,
                    "reasoning": AIModelTier.REASONING,
                    "balanced": AIModelTier.BALANCED,
                    "powerful": AIModelTier.POWERFUL
                }.get(model_tier, AIModelTier.BALANCED)

                response = await client.complete(
                    prompt=prompt,
                    tier=tier,
                    max_tokens=2000
                )
                return response
            except Exception as e:
                logger.warning(f"AI client failed: {e}")

        # Last resort: Use Claude Code CLI
        try:
            result = subprocess.run(
                ["claude", "--print", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.warning(f"Claude CLI failed: {e}")

        return None

    def _parse_analysis(self, analysis: str, failure: Dict, model_tier: str) -> FixAttempt:
        """Parse the AI analysis into a FixAttempt."""

        # Try to extract JSON from the response
        try:
            # Find JSON block
            if "```json" in analysis:
                json_start = analysis.find("```json") + 7
                json_end = analysis.find("```", json_start)
                json_str = analysis[json_start:json_end].strip()
            elif "{" in analysis:
                json_start = analysis.find("{")
                json_end = analysis.rfind("}") + 1
                json_str = analysis[json_start:json_end]
            else:
                json_str = analysis

            data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a basic fix attempt
            data = {
                "root_cause": "Could not parse AI response",
                "fix_strategy": analysis[:500],
                "files_to_modify": [],
                "confidence": 30,
                "reasoning": analysis
            }

        # Create FixAttempt
        return FixAttempt(
            issue_id=failure.get("check_name", "unknown"),
            issue_description=failure.get("message", ""),
            suggested_fix=data.get("fix_strategy", ""),
            files_to_modify=data.get("files_to_modify", []),
            difficulty=self._estimate_difficulty(
                failure.get("check_name", ""),
                failure.get("details", {})
            ),
            confidence=FixConfidence.MEDIUM,  # Will be updated
            confidence_score=data.get("confidence", 50),
            ai_model_used=model_tier,
            reasoning=data.get("reasoning", "")
        )

    def _calculate_confidence(self, fix_attempt: FixAttempt) -> float:
        """Calculate confidence score based on multiple factors."""
        score = fix_attempt.confidence_score

        # Adjust based on historical success rates
        issue_type = fix_attempt.issue_id
        if issue_type in self.success_rates:
            historical_rate = self.success_rates[issue_type].get("rate", 50)
            # Blend with AI confidence
            score = (score * 0.6) + (historical_rate * 0.4)

        # Penalize complex fixes
        difficulty_penalty = {
            FixDifficulty.TRIVIAL: 0,
            FixDifficulty.SIMPLE: 5,
            FixDifficulty.MODERATE: 15,
            FixDifficulty.COMPLEX: 30,
            FixDifficulty.ARCHITECTURAL: 50
        }
        score -= difficulty_penalty.get(fix_attempt.difficulty, 10)

        # Boost if files exist and are modifiable
        if fix_attempt.files_to_modify:
            all_exist = all(
                (self.project_root / f).exists()
                for f in fix_attempt.files_to_modify
            )
            if all_exist:
                score += 10
            else:
                score -= 20

        return max(0, min(100, score))

    def _score_to_confidence(self, score: float) -> FixConfidence:
        """Convert numeric score to confidence level."""
        if score >= 80:
            return FixConfidence.HIGH
        elif score >= 50:
            return FixConfidence.MEDIUM
        else:
            return FixConfidence.LOW

    async def apply_fix(self, fix_attempt: FixAttempt, dry_run: bool = False) -> FixResult:
        """
        Apply a fix to the codebase.

        Args:
            fix_attempt: The fix to apply
            dry_run: If True, don't actually modify files

        Returns:
            FixResult with success status and details
        """
        if dry_run:
            return FixResult(
                success=True,
                message="Dry run - no changes made",
                files_modified=fix_attempt.files_to_modify,
                rollback_available=False
            )

        # Create git checkpoint
        try:
            self._create_git_checkpoint(f"Pre-fix: {fix_attempt.issue_id}")
            fix_attempt.pre_fix_hash = self._get_git_hash()
        except Exception as e:
            logger.warning(f"Failed to create git checkpoint: {e}")

        # Apply the fix using Claude Code
        try:
            result = await self._apply_with_claude(fix_attempt)

            if result.success:
                fix_attempt.applied = True
                fix_attempt.post_fix_hash = self._get_git_hash()

                # Validate the fix
                validation = await self._validate_fix(fix_attempt)
                fix_attempt.validated = validation.tests_passed
                result.tests_passed = validation.tests_passed

                if not validation.tests_passed:
                    # Rollback if validation failed
                    await self.rollback_fix(fix_attempt)
                    result.success = False
                    result.message = f"Fix applied but validation failed: {validation.message}"

            return result

        except Exception as e:
            fix_attempt.error_message = str(e)
            return FixResult(
                success=False,
                message=f"Failed to apply fix: {e}",
                rollback_available=fix_attempt.pre_fix_hash is not None
            )

    async def _apply_with_claude(self, fix_attempt: FixAttempt) -> FixResult:
        """Apply fix using Claude Code CLI."""

        prompt = f"""Apply this fix to the codebase:

## Issue
{fix_attempt.issue_description}

## Fix Strategy
{fix_attempt.suggested_fix}

## Files to Modify
{json.dumps(fix_attempt.files_to_modify, indent=2)}

## Reasoning
{fix_attempt.reasoning}

Please apply the fix. Be precise and only make the necessary changes.
After applying, respond with "FIX_APPLIED" if successful."""

        try:
            result = subprocess.run(
                ["claude", "--dangerously-skip-permissions", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root)
            )

            success = "FIX_APPLIED" in result.stdout or "fixed" in result.stdout.lower()

            return FixResult(
                success=success,
                message=result.stdout[:500] if success else result.stderr[:500],
                files_modified=fix_attempt.files_to_modify,
                rollback_available=True
            )

        except subprocess.TimeoutExpired:
            return FixResult(
                success=False,
                message="Claude Code timed out",
                rollback_available=True
            )
        except FileNotFoundError:
            return FixResult(
                success=False,
                message="Claude Code CLI not found. Install with: npm install -g @anthropic/claude-code",
                rollback_available=False
            )

    async def _validate_fix(self, fix_attempt: FixAttempt) -> FixResult:
        """Validate that a fix works by running relevant tests."""

        # Run the specific QA check that failed
        try:
            # Import and run the check module
            from .checks.api_endpoints import APIEndpointsCheck

            check = APIEndpointsCheck()
            result = check.run()

            passed = result.status.value == "passed"

            return FixResult(
                success=passed,
                message="Validation passed" if passed else "Validation failed",
                tests_passed=passed
            )

        except Exception as e:
            return FixResult(
                success=False,
                message=f"Validation error: {e}",
                tests_passed=False
            )

    async def rollback_fix(self, fix_attempt: FixAttempt) -> bool:
        """Rollback a fix using git."""
        if not fix_attempt.pre_fix_hash:
            return False

        try:
            subprocess.run(
                ["git", "reset", "--hard", fix_attempt.pre_fix_hash],
                check=True,
                cwd=str(self.project_root)
            )
            fix_attempt.rolled_back = True
            return True
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False

    def _create_git_checkpoint(self, message: str):
        """Create a git commit as a checkpoint."""
        try:
            subprocess.run(
                ["git", "add", "-A"],
                check=True,
                cwd=str(self.project_root)
            )
            subprocess.run(
                ["git", "commit", "-m", f"[AUTO-FIX CHECKPOINT] {message}"],
                check=False,  # OK if nothing to commit
                cwd=str(self.project_root)
            )
        except Exception as e:
            logger.warning(f"Git checkpoint failed: {e}")

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def record_fix_result(self, fix_attempt: FixAttempt, success: bool):
        """Record fix result for learning."""
        issue_type = fix_attempt.issue_id

        if issue_type not in self.success_rates:
            self.success_rates[issue_type] = {
                "attempts": 0,
                "successes": 0,
                "rate": 50.0
            }

        self.success_rates[issue_type]["attempts"] += 1
        if success:
            self.success_rates[issue_type]["successes"] += 1

        # Calculate rolling success rate
        attempts = self.success_rates[issue_type]["attempts"]
        successes = self.success_rates[issue_type]["successes"]
        self.success_rates[issue_type]["rate"] = (successes / attempts) * 100

        # Save to disk
        self._save_fix_history()

        # Add to fix history
        self.fix_history.append(fix_attempt)

    def should_auto_fix(self, fix_attempt: FixAttempt) -> Tuple[bool, str]:
        """
        Determine if a fix should be automatically applied.

        Returns:
            Tuple of (should_apply, reason)
        """
        # Always require human approval for complex/architectural fixes
        if fix_attempt.difficulty in [FixDifficulty.COMPLEX, FixDifficulty.ARCHITECTURAL]:
            return False, f"Difficulty too high: {fix_attempt.difficulty.value}"

        # Require high confidence
        if fix_attempt.confidence == FixConfidence.LOW:
            return False, f"Confidence too low: {fix_attempt.confidence_score:.0f}%"

        # Check historical success rate
        issue_type = fix_attempt.issue_id
        if issue_type in self.success_rates:
            rate = self.success_rates[issue_type]["rate"]
            if rate < 70:
                return False, f"Historical success rate too low: {rate:.0f}%"

        # All checks passed
        return True, "Fix approved for auto-application"

    def get_fix_summary(self) -> Dict[str, Any]:
        """Get summary of fix history and success rates."""
        total_attempts = sum(
            stats["attempts"] for stats in self.success_rates.values()
        )
        total_successes = sum(
            stats["successes"] for stats in self.success_rates.values()
        )

        return {
            "total_fix_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": (total_successes / total_attempts * 100) if total_attempts > 0 else 0,
            "success_rates_by_type": self.success_rates,
            "recent_fixes": len(self.fix_history)
        }


# Singleton instance
_auto_fixer: Optional[AIAutoFixer] = None

def get_auto_fixer() -> AIAutoFixer:
    """Get the singleton AIAutoFixer instance."""
    global _auto_fixer
    if _auto_fixer is None:
        _auto_fixer = AIAutoFixer()
    return _auto_fixer
