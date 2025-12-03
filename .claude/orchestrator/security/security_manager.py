"""
Security Manager for Orchestrator
Input validation, PII detection, code scanning
100% Local using free tools (Bandit, Semgrep, Presidio)
"""
from typing import Dict, Any, List, Optional
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Local security management
    - Input validation
    - PII detection
    - Code scanning
    - Rate limiting
    """

    def __init__(self) -> None:
        self.pii_patterns = self._load_pii_patterns()
        self.injection_patterns = self._load_injection_patterns()
        self.rate_limits = {}  # In-memory rate limiting
        logger.info("Security manager initialized")

    def _load_pii_patterns(self) -> Dict[str, str]:
        """Load PII detection patterns"""
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "api_key": r'\b[A-Za-z0-9_-]{32,}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }

    def _load_injection_patterns(self) -> Dict[str, str]:
        """Load injection attack patterns"""
        return {
            "sql_injection": r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
            "command_injection": r"[;&|`$]",
            "path_traversal": r"\.\./",
            "xss": r"<script|javascript:|onerror=|onload="
        }

    def validate_input(self, input_text: str, context: str = "general") -> Dict[str, Any]:
        """Validate user input for security issues"""
        issues = []

        # Check for injection patterns
        for pattern_name, pattern in self.injection_patterns.items():
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append({
                    "type": "injection_attempt",
                    "pattern": pattern_name,
                    "severity": "high"
                })

        # Check for suspicious patterns
        if len(input_text) > 10000:
            issues.append({
                "type": "excessive_length",
                "severity": "medium"
            })

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def detect_pii(self, text: str, redact: bool = False) -> Dict[str, Any]:
        """Detect PII in text"""
        detections = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        if redact and detections:
            redacted_text = text
            for detection in sorted(detections, key=lambda x: x['start'], reverse=True):
                redacted_text = (
                    redacted_text[:detection['start']] +
                    f"[REDACTED-{detection['type'].upper()}]" +
                    redacted_text[detection['end']:]
                )
            return {
                "pii_found": True,
                "count": len(detections),
                "types": list(set(d['type'] for d in detections)),
                "redacted_text": redacted_text
            }

        return {
            "pii_found": len(detections) > 0,
            "count": len(detections),
            "types": list(set(d['type'] for d in detections)),
            "detections": detections
        }

    def scan_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Scan code for security issues
        Uses pattern matching (basic) - can be enhanced with Bandit/Semgrep
        """
        issues = []

        if language == "python":
            # Check for dangerous patterns
            dangerous_patterns = {
                "eval": r"\beval\(",
                "exec": r"\bexec\(",
                "system": r"\bos\.system\(",
                "pickle": r"\bpickle\.loads?\(",
                "sql_string_concat": r"['\"].*\+.*SELECT",
            }

            for issue_type, pattern in dangerous_patterns.items():
                if re.search(pattern, code):
                    issues.append({
                        "type": issue_type,
                        "severity": "high",
                        "message": f"Potentially dangerous: {issue_type}"
                    })

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "language": language
        }

    def check_rate_limit(self, identifier: str, limit: int = 100,
                        window_seconds: int = 3600) -> bool:
        """
        Check rate limit (in-memory, simple implementation)
        Returns True if within limit, False if exceeded
        """
        from datetime import datetime

        now = datetime.now()
        key = f"{identifier}:{window_seconds}"

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Remove old entries
        cutoff = now.timestamp() - window_seconds
        self.rate_limits[key] = [
            ts for ts in self.rate_limits[key]
            if ts > cutoff
        ]

        # Check limit
        if len(self.rate_limits[key]) >= limit:
            return False

        # Add current request
        self.rate_limits[key].append(now.timestamp())
        return True

    def sanitize_output(self, output: str) -> str:
        """Sanitize output to remove sensitive data"""
        result = self.detect_pii(output, redact=True)
        return result.get("redacted_text", output)


# Singleton
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get singleton security manager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


if __name__ == "__main__":
    # Test security manager
    import json

    security = get_security_manager()

    # Test input validation
    test_input = "SELECT * FROM users WHERE id = 1"
    validation = security.validate_input(test_input)
    print(f"\nInput validation: {json.dumps(validation, indent=2)}")

    # Test PII detection
    test_text = "Contact me at john.doe@example.com or call 555-123-4567"
    pii_result = security.detect_pii(test_text, redact=True)
    print(f"\nPII detection: {json.dumps(pii_result, indent=2)}")

    # Test code scanning
    test_code = "result = eval(user_input)"
    scan_result = security.scan_code(test_code)
    print(f"\nCode scan: {json.dumps(scan_result, indent=2)}")

    # Test rate limiting
    for i in range(5):
        allowed = security.check_rate_limit("test-user", limit=3)
        print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")

    print("\nSecurity manager test complete!")
