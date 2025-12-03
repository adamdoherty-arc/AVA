"""
Automation AI Analyzer Service
==============================

AI-powered analysis for automation health, failure patterns, and recommendations.

Features:
- Failure root cause analysis using LLM
- Predictive health monitoring
- Performance optimization recommendations
- Anomaly detection in execution patterns
- Natural language queries about automations

Author: AVA Trading Platform
Created: 2025-11-28
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache

from src.services.llm_service import get_llm_service
from src.services.automation_control_service import get_automation_control_service

logger = logging.getLogger(__name__)


class AutomationAIAnalyzer:
    """
    AI-powered analyzer for automation health and performance.

    Uses LLM to provide intelligent insights about:
    - Root cause analysis for failures
    - Performance optimization suggestions
    - Health predictions
    - Anomaly detection
    """

    def __init__(self):
        """Initialize the AI analyzer."""
        self.llm = get_llm_service()
        self.control = get_automation_control_service()

    def analyze_failure(
        self,
        automation_name: str,
        error_message: str,
        error_traceback: Optional[str] = None,
        recent_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a failure and provide root cause analysis.

        Args:
            automation_name: Name of the failed automation
            error_message: The error message
            error_traceback: Full traceback if available
            recent_history: Recent execution history for context

        Returns:
            Dict with analysis results
        """
        try:
            # Get automation details
            automation = self.control.get_automation(automation_name)

            # Build context for analysis
            context = {
                "automation": {
                    "name": automation_name,
                    "display_name": automation.get('display_name') if automation else automation_name,
                    "description": automation.get('description') if automation else 'Unknown',
                    "schedule": automation.get('schedule_display') if automation else 'Unknown',
                    "category": automation.get('category') if automation else 'Unknown'
                },
                "error": {
                    "message": error_message,
                    "traceback": error_traceback[:2000] if error_traceback else None
                },
                "history": [
                    {
                        "status": h.get('status'),
                        "started_at": str(h.get('started_at')),
                        "duration_seconds": h.get('duration_seconds'),
                        "error_message": h.get('error_message')
                    }
                    for h in (recent_history or [])[:10]
                ]
            }

            prompt = f"""You are an expert DevOps engineer analyzing a failed automation task.

**Automation Details:**
- Name: {context['automation']['name']}
- Description: {context['automation']['description']}
- Schedule: {context['automation']['schedule']}
- Category: {context['automation']['category']}

**Error:**
```
{context['error']['message']}
```

**Traceback (partial):**
```
{context['error']['traceback'] or 'Not available'}
```

**Recent Execution History:**
{json.dumps(context['history'], indent=2)}

**Analyze this failure and provide:**

1. **Root Cause** (1-2 sentences): What is the most likely cause of this failure?

2. **Severity** (Critical/High/Medium/Low): How severe is this issue?

3. **Immediate Fix** (1-2 sentences): What quick action can resolve this?

4. **Long-term Solution** (1-2 sentences): What prevents this in the future?

5. **Related Systems**: What other automations/services might be affected?

Respond in JSON format:
{{
    "root_cause": "string",
    "severity": "Critical|High|Medium|Low",
    "immediate_fix": "string",
    "long_term_solution": "string",
    "related_systems": ["string"],
    "confidence": 0.0-1.0
}}"""

            # Call LLM for analysis
            result = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                use_cache=False  # Don't cache failure analyses
            )

            # Parse response
            response_text = result.get('text', '{}')

            # Extract JSON from response
            try:
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    analysis = json.loads(response_text[start:end])
                else:
                    analysis = {"error": "Could not parse analysis"}
            except json.JSONDecodeError:
                analysis = {"error": "Could not parse analysis", "raw": response_text[:500]}

            return {
                "status": "success",
                "automation_name": automation_name,
                "analysis": analysis,
                "llm_provider": result.get('provider'),
                "llm_model": result.get('model'),
                "cost": result.get('cost', 0)
            }

        except Exception as e:
            logger.error(f"Error analyzing failure for {automation_name}: {e}")
            return {
                "status": "error",
                "automation_name": automation_name,
                "error": str(e)
            }

    def get_health_prediction(
        self,
        automation_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Predict automation health based on historical patterns.

        Args:
            automation_name: Specific automation (None for all)
            hours: Historical window to analyze

        Returns:
            Health predictions and risk scores
        """
        try:
            # Get execution history
            history = self.control.get_execution_history(
                automation_name=automation_name,
                limit=200,
                since=datetime.now() - timedelta(hours=hours * 3)  # 3x window for patterns
            )

            if not history:
                return {
                    "status": "success",
                    "prediction": "insufficient_data",
                    "message": "Not enough historical data for prediction"
                }

            # Calculate metrics
            total = len(history)
            successful = len([h for h in history if h.get('status') == 'success'])
            failed = len([h for h in history if h.get('status') == 'failed'])
            success_rate = (successful / total * 100) if total > 0 else 0

            # Calculate trend (last 12h vs previous 12h)
            now = datetime.now()
            recent = [h for h in history if h.get('started_at') and
                     (now - h['started_at']).total_seconds() < hours * 3600 / 2]
            older = [h for h in history if h.get('started_at') and
                    (now - h['started_at']).total_seconds() >= hours * 3600 / 2]

            recent_success = len([h for h in recent if h.get('status') == 'success']) / max(len(recent), 1)
            older_success = len([h for h in older if h.get('status') == 'success']) / max(len(older), 1)

            trend = "improving" if recent_success > older_success else \
                    "declining" if recent_success < older_success else "stable"

            # Calculate risk score (0-100, higher = more risky)
            risk_score = int(100 - success_rate)
            if trend == "declining":
                risk_score = min(100, risk_score + 15)

            # Failure patterns
            failure_messages = [h.get('error_message', '') for h in history if h.get('status') == 'failed']
            unique_errors = list(set([m for m in failure_messages if m]))[:5]

            # Generate AI insights if risk is elevated
            ai_insights = None
            if risk_score > 30 and unique_errors:
                ai_insights = self._generate_health_insights(
                    automation_name,
                    success_rate,
                    trend,
                    unique_errors
                )

            return {
                "status": "success",
                "automation_name": automation_name or "all",
                "metrics": {
                    "total_executions": total,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": round(success_rate, 1),
                    "trend": trend,
                    "risk_score": risk_score
                },
                "prediction": {
                    "health_status": "healthy" if risk_score < 20 else \
                                    "warning" if risk_score < 50 else "critical",
                    "next_24h_failure_risk": "low" if risk_score < 20 else \
                                             "medium" if risk_score < 50 else "high"
                },
                "common_errors": unique_errors,
                "ai_insights": ai_insights,
                "time_window_hours": hours
            }

        except Exception as e:
            logger.error(f"Error generating health prediction: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_health_insights(
        self,
        automation_name: Optional[str],
        success_rate: float,
        trend: str,
        errors: List[str]
    ) -> Optional[Dict]:
        """Generate AI insights for health issues."""
        try:
            prompt = f"""You are a DevOps AI assistant analyzing automation health.

**Automation:** {automation_name or 'All automations'}
**Success Rate:** {success_rate:.1f}%
**Trend:** {trend}

**Recent Error Patterns:**
{chr(10).join(f'- {e[:200]}' for e in errors[:5])}

Provide brief recommendations in JSON format:
{{
    "summary": "1 sentence overall assessment",
    "recommendations": ["action 1", "action 2", "action 3"],
    "priority_action": "single most important action"
}}"""

            result = self.llm.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
                use_cache=True
            )

            response_text = result.get('text', '{}')
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start >= 0 and end > start:
                return json.loads(response_text[start:end])

        except Exception as e:
            logger.warning(f"Could not generate AI health insights: {e}")

        return None

    def get_optimization_recommendations(
        self,
        automation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered optimization recommendations.

        Args:
            automation_name: Specific automation (None for all)

        Returns:
            Optimization recommendations
        """
        try:
            # Get automations
            if automation_name:
                automation = self.control.get_automation(automation_name)
                automations = [automation] if automation else []
            else:
                automations = self.control.get_all_automations()

            if not automations:
                return {
                    "status": "success",
                    "recommendations": [],
                    "message": "No automations found"
                }

            # Analyze each automation
            recommendations = []

            for auto in automations:
                rec = self._analyze_automation_performance(auto)
                if rec:
                    recommendations.append(rec)

            # Sort by priority
            recommendations.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

            return {
                "status": "success",
                "recommendations": recommendations[:10],  # Top 10
                "total_analyzed": len(automations)
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _analyze_automation_performance(self, automation: Dict) -> Optional[Dict]:
        """Analyze single automation performance and generate recommendations."""
        name = automation.get('name')
        success_rate = automation.get('success_rate_24h')
        avg_duration = automation.get('avg_duration_24h')
        total_runs = automation.get('total_runs_24h', 0)
        last_error = automation.get('last_error')

        issues = []
        priority_score = 0

        # Check success rate
        if success_rate is not None and success_rate < 90:
            issues.append({
                "type": "low_success_rate",
                "message": f"Success rate ({success_rate:.1f}%) is below 90% threshold",
                "severity": "high" if success_rate < 70 else "medium"
            })
            priority_score += 30 if success_rate < 70 else 15

        # Check for recent errors
        if last_error:
            issues.append({
                "type": "recent_failure",
                "message": f"Recent failure: {last_error[:100]}",
                "severity": "high"
            })
            priority_score += 25

        # Check execution duration
        if avg_duration and avg_duration > 60:  # Over 1 minute
            issues.append({
                "type": "slow_execution",
                "message": f"Average duration ({avg_duration:.1f}s) may be too long",
                "severity": "low"
            })
            priority_score += 10

        # Check if disabled
        if not automation.get('is_enabled'):
            issues.append({
                "type": "disabled",
                "message": "Automation is currently disabled",
                "severity": "info"
            })
            priority_score += 5

        if not issues:
            return None

        return {
            "automation_name": name,
            "display_name": automation.get('display_name'),
            "category": automation.get('category'),
            "issues": issues,
            "priority_score": priority_score,
            "metrics": {
                "success_rate_24h": success_rate,
                "avg_duration_24h": avg_duration,
                "total_runs_24h": total_runs
            }
        }

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer natural language questions about automations.

        Args:
            question: Natural language question

        Returns:
            AI-generated answer with supporting data
        """
        try:
            # Gather context
            dashboard_stats = self.control.get_dashboard_stats(hours=24)
            all_automations = self.control.get_all_automations()
            recent_failures = self.control.get_execution_history(status='failed', limit=10)

            # Build context string
            context = f"""
**System Overview (Last 24h):**
- Total Automations: {dashboard_stats.get('automations', {}).get('total', 0)}
- Enabled: {dashboard_stats.get('automations', {}).get('enabled', 0)}
- Disabled: {dashboard_stats.get('automations', {}).get('disabled', 0)}
- Total Executions: {dashboard_stats.get('executions', {}).get('total_executions', 0)}
- Success Rate: {dashboard_stats.get('executions', {}).get('success_rate', 'N/A')}%
- Failed: {dashboard_stats.get('executions', {}).get('failed', 0)}
- Running: {dashboard_stats.get('executions', {}).get('running', 0)}

**Automations by Category:**
{json.dumps([{
    'name': a['name'],
    'category': a['category'],
    'enabled': a['is_enabled'],
    'success_rate': a.get('success_rate_24h'),
    'last_status': a.get('last_run_status')
} for a in all_automations], indent=2)}

**Recent Failures:**
{json.dumps([{
    'name': f.get('automation_name'),
    'error': f.get('error_message', '')[:100],
    'when': str(f.get('started_at'))
} for f in recent_failures[:5]], indent=2)}
"""

            prompt = f"""You are AVA, an AI assistant for the AVA Trading Platform automation system.

{context}

**User Question:** {question}

Provide a helpful, concise answer based on the data above. If the question asks about specific automations, refer to them by name. Include relevant metrics when helpful. Keep the response under 200 words."""

            result = self.llm.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.5,
                use_cache=True
            )

            return {
                "status": "success",
                "question": question,
                "answer": result.get('text', 'Unable to generate answer'),
                "llm_provider": result.get('provider'),
                "cost": result.get('cost', 0)
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "status": "error",
                "question": question,
                "error": str(e)
            }


# Global singleton
_analyzer: Optional[AutomationAIAnalyzer] = None


def get_automation_ai_analyzer() -> AutomationAIAnalyzer:
    """Get the global AI analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AutomationAIAnalyzer()
    return _analyzer
