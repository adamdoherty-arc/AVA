"""
QA Dashboard API Routes

Provides API endpoints to monitor and manage the Continuous QA system.
Supports both file-based logging and PostgreSQL database storage.
"""

import json
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".claude" / "orchestrator"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import database manager
try:
    from src.qa_issues_db_manager import QAIssuesDBManager
    qa_db = QAIssuesDBManager()
    if qa_db.check_schema_exists():
        DB_AVAILABLE = True
        logger.info("QA Dashboard: Database connection established")
    else:
        DB_AVAILABLE = False
        qa_db = None
        logger.warning("QA Dashboard: Database schema not found, using file-based storage")
except Exception as e:
    DB_AVAILABLE = False
    qa_db = None
    logger.warning(f"QA Dashboard: Database not available ({e}), using file-based storage")

router = APIRouter(prefix="/api/qa", tags=["QA Dashboard"])

# Paths to QA data files
QA_DIR = PROJECT_ROOT / ".claude" / "orchestrator" / "continuous_qa"
LOGS_DIR = QA_DIR / "logs"
DATA_DIR = QA_DIR / "data"

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class QAStatus(BaseModel):
    """Current QA system status."""
    is_running: bool
    last_run_time: Optional[str]
    next_run_time: Optional[str]
    health_score: float
    total_cycles: int
    total_issues_found: int
    total_issues_fixed: int


class Accomplishment(BaseModel):
    """Single accomplishment entry."""
    timestamp: str
    category: str
    module: str
    message: str
    severity: str
    files_affected: List[str]


class HotSpot(BaseModel):
    """File with frequent issues."""
    file: str
    issue_count: int
    severity_score: int
    patterns: List[str]


class IssueStatusUpdate(BaseModel):
    """Request body for updating issue status."""
    status: str
    resolved_by: Optional[str] = None
    notes: Optional[str] = None


@router.get("/status", response_model=Dict[str, Any])
async def get_qa_status():
    """Get current QA system status."""
    # Check both locations for status.json
    status_file = DATA_DIR / "status.json"
    if not status_file.exists():
        status_file = LOGS_DIR / "status.json"

    default_status = {
        "is_running": False,
        "last_run_time": None,
        "next_run_time": None,
        "health_score": 0,
        "total_cycles": 0,
        "total_issues_found": 0,
        "total_issues_fixed": 0,
        "last_run_duration_seconds": 0,
        "critical_failures": 0,
    }

    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
                # Map fields from actual file to expected format
                mapped_status = {
                    "is_running": status.get("is_currently_running", False),
                    "last_run_time": status.get("last_run"),
                    "next_run_time": status.get("next_run"),
                    "health_score": status.get("health_score", 0),
                    "total_cycles": status.get("current_cycle", 0),
                    "total_issues_found": status.get("last_accomplishments_count", 0),
                    "total_issues_fixed": status.get("last_accomplishments_count", 0),
                    "last_run_duration_seconds": 0,
                    "critical_failures": 0,
                    "run_id": status.get("last_run_id"),
                }
                return {**default_status, **mapped_status}
        except Exception as e:
            return {**default_status, "error": str(e)}

    return default_status


@router.get("/health-history")
async def get_health_history(days: int = 7):
    """Get health score history."""
    # Check both locations
    health_file = DATA_DIR / "health_history.jsonl"
    if not health_file.exists():
        health_file = LOGS_DIR / "health_history.jsonl"

    history = []
    if health_file.exists():
        try:
            with open(health_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        history.append(entry)
        except Exception as e:
            return {"error": str(e), "history": []}

    # Return last N days worth
    return {"history": history[-days * 24:]}  # Assuming ~hourly runs


@router.get("/accomplishments")
async def get_accomplishments(limit: int = 50):
    """Get recent accomplishments."""
    # Check both locations
    accomplishments_file = DATA_DIR / "accomplishments.jsonl"
    if not accomplishments_file.exists():
        accomplishments_file = LOGS_DIR / "accomplishments.jsonl"

    accomplishments = []
    if accomplishments_file.exists():
        try:
            with open(accomplishments_file, 'r') as f:
                lines = f.readlines()
                # Get last N lines
                for line in lines[-limit:]:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            accomplishments.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return {"error": str(e), "accomplishments": []}

    # Reverse to show newest first
    accomplishments.reverse()

    # Group by category
    by_category = {}
    for acc in accomplishments:
        cat = acc.get('category', 'other')
        if cat not in by_category:
            by_category[cat] = 0
        by_category[cat] += 1

    return {
        "accomplishments": accomplishments,
        "by_category": by_category,
        "total": len(accomplishments),
    }


@router.get("/patterns")
async def get_learned_patterns():
    """Get learned patterns and statistics."""
    patterns_file = DATA_DIR / "learned_patterns.json"

    if not patterns_file.exists():
        return {"patterns": [], "statistics": {}}

    try:
        with open(patterns_file, 'r') as f:
            data = json.load(f)

        patterns = data.get('patterns', [])

        # Calculate statistics
        by_severity = {}
        by_category = {}
        total_occurrences = 0

        for p in patterns:
            sev = p.get('severity', 'low')
            cat = p.get('category', 'other')
            occ = p.get('occurrences', 0)

            by_severity[sev] = by_severity.get(sev, 0) + occ
            by_category[cat] = by_category.get(cat, 0) + occ
            total_occurrences += occ

        return {
            "patterns": patterns[:20],  # Top 20
            "statistics": {
                "total_patterns": len(patterns),
                "total_occurrences": total_occurrences,
                "by_severity": by_severity,
                "by_category": by_category,
            }
        }
    except Exception as e:
        return {"error": str(e), "patterns": []}


@router.get("/hot-spots")
async def get_hot_spots():
    """Get files that frequently have issues."""
    patterns_file = DATA_DIR / "learned_patterns.json"

    if not patterns_file.exists():
        return {"hot_spots": []}

    try:
        with open(patterns_file, 'r') as f:
            data = json.load(f)

        # Aggregate files across patterns
        file_issues = {}
        severity_weights = {'low': 1, 'medium': 2, 'high': 5, 'critical': 10}

        for pattern in data.get('patterns', []):
            weight = severity_weights.get(pattern.get('severity', 'low'), 1)
            for file_path in pattern.get('files_affected', []):
                if file_path not in file_issues:
                    file_issues[file_path] = {
                        'count': 0,
                        'severity_score': 0,
                        'patterns': set()
                    }
                file_issues[file_path]['count'] += pattern.get('occurrences', 1)
                file_issues[file_path]['severity_score'] += weight
                file_issues[file_path]['patterns'].add(pattern.get('id', 'unknown'))

        # Convert to list and sort
        hot_spots = [
            {
                'file': f,
                'issue_count': data['count'],
                'severity_score': data['severity_score'],
                'patterns': list(data['patterns'])[:5],
            }
            for f, data in file_issues.items()
        ]

        hot_spots.sort(key=lambda x: x['severity_score'], reverse=True)

        return {"hot_spots": hot_spots[:20]}

    except Exception as e:
        return {"error": str(e), "hot_spots": []}


@router.get("/enhancements")
async def get_enhancement_log(limit: int = 50):
    """Get recent enhancement log."""
    # Check both locations
    enhancements_file = LOGS_DIR / "enhancements.jsonl"
    if not enhancements_file.exists():
        enhancements_file = DATA_DIR / "enhancements.jsonl"

    enhancements = []
    if enhancements_file.exists():
        try:
            with open(enhancements_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            enhancements.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            return {"error": str(e), "enhancements": []}

    enhancements.reverse()

    # Group by category
    by_category = {}
    for enh in enhancements:
        cat = enh.get('category', 'other')
        if cat not in by_category:
            by_category[cat] = {'total': 0, 'applied': 0}
        by_category[cat]['total'] += 1
        if enh.get('applied'):
            by_category[cat]['applied'] += 1

    return {
        "enhancements": enhancements,
        "by_category": by_category,
        "total": len(enhancements),
    }


@router.post("/run-once")
async def trigger_qa_run(background_tasks: BackgroundTasks):
    """Trigger a single QA run."""
    try:
        from continuous_qa.qa_runner import QARunner

        def run_qa():
            runner = QARunner(visible=False)
            runner.run_once()

        background_tasks.add_task(run_qa)

        return {
            "status": "started",
            "message": "QA run started in background",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_qa_config():
    """Get QA system configuration."""
    config_file = QA_DIR / "config" / "qa_config.yaml"

    if not config_file.exists():
        return {"config": {}, "error": "Config file not found"}

    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return {"config": config}
    except Exception as e:
        return {"error": str(e), "config": {}}


@router.get("/summary")
async def get_qa_summary():
    """Get a comprehensive QA summary for the dashboard."""
    status = await get_qa_status()
    accomplishments = await get_accomplishments(limit=10)
    patterns = await get_learned_patterns()
    hot_spots = await get_hot_spots()

    return {
        "status": status,
        "recent_accomplishments": accomplishments.get('accomplishments', [])[:5],
        "accomplishments_by_category": accomplishments.get('by_category', {}),
        "pattern_statistics": patterns.get('statistics', {}),
        "top_hot_spots": hot_spots.get('hot_spots', [])[:5],
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# DATABASE-BACKED ENDPOINTS (Enhanced Issue Tracking)
# =============================================================================

@router.get("/db/dashboard")
async def get_db_dashboard_summary():
    """Get comprehensive dashboard summary from database."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "use_fallback": True}

    try:
        summary = qa_db.get_dashboard_summary()
        return {
            "database_available": True,
            **summary
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}")
        return {"error": str(e), "use_fallback": True}


@router.get("/db/runs")
async def get_db_runs(limit: int = Query(20, ge=1, le=100)):
    """Get recent QA runs from database."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "runs": []}

    try:
        runs = qa_db.get_recent_runs(limit=limit)
        return {"runs": runs, "total": len(runs)}
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return {"error": str(e), "runs": []}


@router.get("/db/runs/{run_id}")
async def get_db_run_details(run_id: int):
    """Get details for a specific QA run."""
    if not DB_AVAILABLE or not qa_db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        run = qa_db.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Get check results for this run
        check_results = qa_db.get_check_results(run_id)

        # Return run data merged with check_results for frontend compatibility
        return {
            **run,
            "check_results": check_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/db/issues")
async def get_db_issues(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status (open, fixing, fixed, ignored, wont_fix)"),
    limit: int = Query(50, ge=1, le=200)
):
    """Get issues from database with optional filters."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "issues": []}

    try:
        # Use flexible get_issues method that supports all filters
        issues = qa_db.get_issues(
            status_filter=status,
            severity_filter=severity,
            category_filter=category,
            limit=limit
        )

        return {
            "issues": issues,
            "total": len(issues),
            "filters": {
                "severity": severity,
                "category": category,
                "status": status,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching issues: {e}")
        return {"error": str(e), "issues": []}


@router.get("/db/issues/{issue_id}")
async def get_db_issue_details(issue_id: int):
    """Get details for a specific issue including fix history and occurrences."""
    if not DB_AVAILABLE or not qa_db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        issue = qa_db.get_issue(issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")

        # Get fix history and occurrences
        fix_history = qa_db.get_fix_history(issue_id)
        occurrences = qa_db.get_issue_occurrences(issue_id)

        # Return issue data merged with related data for frontend compatibility
        return {
            **issue,
            "occurrences": occurrences,
            "fixes": fix_history,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching issue details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/db/issues/{issue_id}/status")
async def update_db_issue_status(
    issue_id: int,
    body: IssueStatusUpdate
):
    """Update issue status."""
    if not DB_AVAILABLE or not qa_db:
        raise HTTPException(status_code=503, detail="Database not available")

    valid_statuses = ["open", "fixing", "fixed", "ignored", "wont_fix"]
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    try:
        success = qa_db.update_issue_status(
            issue_id=issue_id,
            status=body.status,
            resolved_by=body.resolved_by,
            resolution_notes=body.notes
        )

        if not success:
            raise HTTPException(status_code=404, detail="Issue not found")

        return {
            "success": True,
            "issue_id": issue_id,
            "new_status": body.status,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating issue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/db/health-history")
async def get_db_health_history(days: int = Query(7, ge=1, le=30)):
    """Get health score history from database."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "history": []}

    try:
        history = qa_db.get_health_history(days=days)
        return {"history": history, "days": days}
    except Exception as e:
        logger.error(f"Error fetching health history: {e}")
        return {"error": str(e), "history": []}


@router.get("/db/health-trend")
async def get_db_health_trend(hours: int = Query(48, ge=1, le=168)):
    """Get hourly health score trend."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "trend": []}

    try:
        trend = qa_db.get_health_trend(hours=hours)
        return {"trend": trend}
    except Exception as e:
        logger.error(f"Error fetching health trend: {e}")
        return {"error": str(e), "trend": []}


@router.get("/db/hot-spots")
async def get_db_hot_spots(limit: int = Query(20, ge=1, le=50)):
    """Get hot spots (files with frequent issues) from database."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "hot_spots": []}

    try:
        hot_spots = qa_db.get_hot_spots(limit=limit)
        return {"hot_spots": hot_spots}
    except Exception as e:
        logger.error(f"Error fetching hot spots: {e}")
        return {"error": str(e), "hot_spots": []}


@router.get("/db/issue-trends")
async def get_db_issue_trends():
    """Get issue trends by category."""
    if not DB_AVAILABLE or not qa_db:
        return {"error": "Database not available", "trends": []}

    try:
        trends = qa_db.get_issue_trends_by_category()
        return {"trends": trends}
    except Exception as e:
        logger.error(f"Error fetching issue trends: {e}")
        return {"error": str(e), "trends": []}


# =============================================================================
# AI-POWERED ANALYSIS ENDPOINTS
# =============================================================================

# Import LLM service for real AI analysis
LLM_AVAILABLE = False
llm_service = None
try:
    from src.services.llm_service import get_llm_service
    llm_service = get_llm_service()
    if llm_service.get_available_providers():
        LLM_AVAILABLE = True
        logger.info(f"AI Analysis: LLM service available with providers: {llm_service.get_available_providers()}")
except Exception as e:
    logger.warning(f"AI Analysis: LLM service not available ({e}), using rule-based fallback")


class AIAnalysisRequest(BaseModel):
    """Request body for AI analysis."""
    prompt: str
    context: Optional[Dict[str, Any]] = None
    use_real_ai: bool = True  # Whether to use real LLM or rule-based


class AIAnalysisResponse(BaseModel):
    """Response from AI analysis."""
    response: str
    suggestions: Optional[List[str]] = None
    priority_issues: Optional[List[int]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    ai_powered: bool = False


async def generate_ai_analysis_with_llm(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI analysis using real LLM service.

    Args:
        prompt: User's question
        context: QA context including issues summary

    Returns:
        Dict with response and metadata
    """
    if not LLM_AVAILABLE or not llm_service:
        # Fall back to rule-based
        return {
            "response": generate_rule_based_analysis(prompt, context),
            "ai_powered": False,
            "provider": None,
            "model": None
        }

    try:
        # Build a comprehensive prompt for the LLM
        issues_summary = context.get('issues_summary', [])
        open_issues = context.get('open_issues', 0)
        critical_issues = context.get('critical_issues', 0)

        system_context = f"""You are an AI QA Assistant analyzing software quality issues.

Current QA Status:
- Total Open Issues: {open_issues}
- Critical Issues: {critical_issues}

Issues Summary (top 10):
"""
        for issue in issues_summary[:10]:
            system_context += f"- [{issue.get('severity', 'unknown').upper()}] {issue.get('title', 'Unknown')} (Module: {issue.get('module', 'unknown')}, Status: {issue.get('status', 'unknown')})\n"

        system_context += """
Please analyze the QA situation and respond to the user's question.
Be concise, actionable, and prioritize by severity.
Use markdown formatting for readability.
"""

        full_prompt = f"{system_context}\n\nUser Question: {prompt}"

        # Generate response using LLM service
        result = llm_service.generate(
            prompt=full_prompt,
            max_tokens=1000,
            temperature=0.7,
            use_cache=True
        )

        return {
            "response": result['text'],
            "ai_powered": True,
            "provider": result['provider'],
            "model": result['model'],
            "cost": result.get('cost', 0),
            "cached": result.get('cached', False)
        }

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fall back to rule-based
        return {
            "response": generate_rule_based_analysis(prompt, context),
            "ai_powered": False,
            "provider": None,
            "model": None,
            "error": str(e)
        }


def generate_rule_based_analysis(prompt: str, context: Dict[str, Any]) -> str:
    """
    Generate AI-powered analysis based on issue context.
    Uses rule-based analysis with smart heuristics.
    Can be extended to use actual LLM API calls.
    """
    prompt_lower = prompt.lower()
    issues_summary = context.get('issues_summary', [])
    open_issues = context.get('open_issues', 0)
    critical_issues = context.get('critical_issues', 0)

    # Summary request
    if any(word in prompt_lower for word in ['summary', 'overview', 'status']):
        critical_list = [i for i in issues_summary if i.get('severity') == 'critical']
        high_list = [i for i in issues_summary if i.get('severity') == 'high']

        response = f"""**QA System Summary**

**Current Status:**
- Total Open Issues: {open_issues}
- Critical Issues: {critical_issues}
- High Priority Issues: {len(high_list)}

"""
        if critical_issues > 0:
            response += f"""**Critical Issues Requiring Immediate Attention:**
{chr(10).join([f'- {i["title"]} ({i["module"]})' for i in critical_list[:5]])}

**Recommendation:** Focus on critical issues first. These may impact system stability or user experience.
"""
        elif open_issues > 10:
            response += """**Status:** Multiple issues detected. Consider scheduling a focused debugging session.

**Recommendation:** Prioritize by severity, then by module impact.
"""
        else:
            response += """**Status:** System is in good health with minimal issues.

**Recommendation:** Continue regular maintenance and monitoring.
"""
        return response

    # Critical issues request
    if any(word in prompt_lower for word in ['critical', 'urgent', 'important', 'emergency']):
        critical_list = [i for i in issues_summary if i.get('severity') == 'critical']

        if critical_list:
            response = f"""**Critical Issues Analysis ({len(critical_list)} found)**

"""
            for i, issue in enumerate(critical_list[:5], 1):
                response += f"""{i}. **{issue['title']}**
   - Module: {issue['module']}
   - Status: {issue['status']}

"""
            response += """**Immediate Actions Recommended:**
1. Review error logs for each critical issue
2. Check for recent deployments or changes
3. Verify database connections and external services
4. Consider rolling back recent changes if issues appeared suddenly
"""
        else:
            response = """**No Critical Issues Found!**

Your system is currently stable with no critical issues requiring immediate attention.

**Preventive Recommendations:**
1. Review warning-level issues before they escalate
2. Run comprehensive test suite regularly
3. Monitor system metrics for early warning signs
"""
        return response

    # Priority/prioritize request
    if any(word in prompt_lower for word in ['prioritize', 'priority', 'order', 'rank']):
        # Sort by severity and occurrence
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_issues = sorted(
            [i for i in issues_summary if i.get('status') in ['open', 'fixing']],
            key=lambda x: severity_order.get(x.get('severity', 'low'), 3)
        )

        response = """**Issue Priority Ranking**

Based on severity, frequency, and impact analysis:

"""
        for i, issue in enumerate(sorted_issues[:10], 1):
            sev = issue.get('severity', 'low').upper()
            response += f"""{i}. **[{sev}]** {issue['title']}
   Module: {issue['module']} | Status: {issue['status']}

"""

        response += """**Prioritization Criteria:**
- Critical: Immediate attention required (blocking issues)
- High: Should be addressed within 24 hours
- Medium: Plan for current sprint
- Low: Can be scheduled for future maintenance
"""
        return response

    # Fix suggestion request
    if any(word in prompt_lower for word in ['fix', 'solve', 'resolve', 'suggest', 'how']):
        open_list = [i for i in issues_summary if i.get('status') in ['open', 'fixing']]

        if open_list:
            issue = open_list[0]
            response = f"""**Fix Suggestion for: {issue['title']}**

**Issue Details:**
- Module: {issue['module']}
- Severity: {issue['severity']}
- Current Status: {issue['status']}

**Suggested Approach:**

1. **Investigate Root Cause:**
   - Check recent changes in the {issue['module']} module
   - Review error logs for stack traces
   - Verify configuration and environment variables

2. **Implement Fix:**
   - Create a failing test that reproduces the issue
   - Apply minimal code changes to fix the problem
   - Ensure backwards compatibility

3. **Validate Solution:**
   - Run the test suite
   - Perform manual testing if needed
   - Review changes with team

4. **Deploy & Monitor:**
   - Deploy to staging first
   - Monitor for regressions
   - Update documentation if needed

**Common Fix Patterns:**
- If import error: Check module paths and dependencies
- If type error: Add proper type checking or validation
- If connection error: Verify service availability and credentials
"""
        else:
            response = """**No Open Issues to Fix!**

All issues are either resolved or in progress. Great job!

**Maintenance Recommendations:**
1. Review recently fixed issues for patterns
2. Update tests to prevent regression
3. Document solutions for future reference
"""
        return response

    # Default helpful response
    return f"""**AI QA Assistant**

I can help you analyze and manage QA issues. Here are some things you can ask:

**Quick Commands:**
- "Give me a summary" - Overview of current QA status
- "Show critical issues" - List urgent problems
- "Prioritize issues" - Get ranked list by importance
- "Suggest a fix" - Get fix recommendations for top issue

**Current Stats:**
- Open Issues: {open_issues}
- Critical Issues: {critical_issues}

What would you like to know about your QA issues?
"""


@router.post("/ai/analyze", response_model=AIAnalysisResponse)
async def analyze_with_ai(request: AIAnalysisRequest):
    """
    AI-powered analysis endpoint for QA issues.

    Features:
    - Real LLM-powered analysis (when available)
    - Intelligent fallback to rule-based analysis
    - Issue summaries and recommendations
    - Priority rankings
    - Fix suggestions
    - Pattern analysis
    """
    try:
        context = request.context or {}

        # Use real AI if requested and available
        if request.use_real_ai and LLM_AVAILABLE:
            result = await generate_ai_analysis_with_llm(request.prompt, context)
        else:
            result = {
                "response": generate_rule_based_analysis(request.prompt, context),
                "ai_powered": False,
                "provider": None,
                "model": None
            }

        # Extract any suggested priority issue IDs from context
        priority_issues = None
        if context and 'issues_summary' in context:
            critical = [
                i for i in context['issues_summary']
                if i.get('severity') == 'critical'
            ]
            if critical:
                priority_issues = []

        # Generate contextual suggestions
        suggestions = None
        if context:
            critical_count = context.get('critical_issues', 0)
            open_count = context.get('open_issues', 0)

            if critical_count > 0:
                suggestions = [
                    f"URGENT: {critical_count} critical issue(s) require immediate attention",
                    "Review error logs for affected modules",
                    "Consider pausing deployments until resolved",
                ]
            elif open_count > 10:
                suggestions = [
                    "Schedule a focused debugging session",
                    "Group similar issues for batch resolution",
                    "Review recent commits for potential root causes",
                ]

        return AIAnalysisResponse(
            response=result["response"],
            suggestions=suggestions,
            priority_issues=priority_issues,
            provider=result.get("provider"),
            model=result.get("model"),
            ai_powered=result.get("ai_powered", False)
        )
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return AIAnalysisResponse(
            response=f"I encountered an error while analyzing: {str(e)}. Please try a simpler query.",
            suggestions=None,
            priority_issues=None,
            ai_powered=False
        )


@router.get("/ai/status")
async def get_ai_status():
    """Get AI service status and available providers."""
    if LLM_AVAILABLE and llm_service:
        try:
            info = llm_service.get_service_info()
            usage = llm_service.get_usage_stats()
            return {
                "available": True,
                "providers": info.get("available_providers", []),
                "total_cost": usage.get("total_cost", 0),
                "cache_enabled": info.get("cache_enabled", False),
                "intelligent_routing": info.get("intelligent_routing_enabled", False)
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    return {
        "available": False,
        "message": "LLM service not initialized"
    }


@router.get("/ai/recommendations")
async def get_ai_recommendations():
    """
    Get AI-generated recommendations based on current QA state.
    """
    if not DB_AVAILABLE or not qa_db:
        return {
            "recommendations": [
                "Database not connected - recommendations limited",
                "Consider setting up PostgreSQL for full QA tracking"
            ],
            "priority": "medium"
        }

    try:
        summary = qa_db.get_dashboard_summary()
        open_issues = summary.get('open_issues', 0)
        by_severity = summary.get('by_severity', {})
        critical_count = by_severity.get('critical', 0)
        high_count = by_severity.get('high', 0)

        recommendations = []
        priority = "low"

        if critical_count > 0:
            priority = "critical"
            recommendations.append(f"URGENT: {critical_count} critical issue(s) need immediate attention")
            recommendations.append("Review critical issues before any new deployments")

        if high_count > 3:
            if priority != "critical":
                priority = "high"
            recommendations.append(f"{high_count} high-priority issues detected - schedule debugging session")

        if open_issues > 20:
            recommendations.append("Consider a focused issue triage session")
            recommendations.append("Look for patterns across issues to identify root causes")

        if open_issues == 0:
            recommendations.append("System is healthy! Consider adding more automated checks")
            recommendations.append("Review recently fixed issues for documentation")

        if not recommendations:
            recommendations = [
                "System is operating normally",
                "Continue regular monitoring",
                "Consider reviewing medium/low priority issues"
            ]

        return {
            "recommendations": recommendations,
            "priority": priority,
            "stats": {
                "open_issues": open_issues,
                "critical": critical_count,
                "high": high_count
            }
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {
            "recommendations": ["Error generating recommendations"],
            "priority": "unknown",
            "error": str(e)
        }


@router.get("/db/status")
async def get_db_status():
    """Check database connectivity status."""
    return {
        "available": DB_AVAILABLE,
        "message": "Database connected" if DB_AVAILABLE else "Database not available",
        "connection_type": "postgresql" if DB_AVAILABLE else "file-based",
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# WEBSOCKET REAL-TIME UPDATES
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time QA updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")

        # Start broadcast task if not running
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._periodic_broadcast())

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn)

    async def _periodic_broadcast(self):
        """Periodically broadcast QA status updates."""
        while self.active_connections:
            try:
                # Gather QA status data
                status_data = await get_qa_status()
                summary_data = None

                if DB_AVAILABLE and qa_db:
                    try:
                        summary_data = qa_db.get_dashboard_summary()
                    except Exception as e:
                        logger.warning(f"Error getting DB summary for broadcast: {e}")

                # Build update message
                update = {
                    "type": "qa_update",
                    "timestamp": datetime.now().isoformat(),
                    "status": status_data,
                    "summary": summary_data,
                    "active_connections": len(self.active_connections)
                }

                await self.broadcast(update)

                # Wait 10 seconds before next update
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic broadcast: {e}")
                await asyncio.sleep(5)


# Global connection manager instance
ws_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time QA updates.

    Clients connect to receive:
    - Periodic status updates (every 10 seconds)
    - Instant notifications for new issues
    - QA run start/complete events
    - Issue status changes

    Message format:
    {
        "type": "qa_update" | "new_issue" | "issue_updated" | "run_started" | "run_completed",
        "timestamp": "ISO datetime",
        "data": {...}
    }
    """
    await ws_manager.connect(websocket)

    # Send initial status on connect
    try:
        status = await get_qa_status()
        await ws_manager.send_personal({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message": "Connected to QA Dashboard WebSocket"
        }, websocket)
    except Exception as e:
        logger.error(f"Error sending initial status: {e}")

    try:
        while True:
            # Wait for messages from client (heartbeat, commands, etc.)
            data = await websocket.receive_json()

            # Handle client messages
            if data.get("type") == "ping":
                await ws_manager.send_personal({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)

            elif data.get("type") == "subscribe":
                # Client can subscribe to specific event types
                await ws_manager.send_personal({
                    "type": "subscribed",
                    "events": data.get("events", ["all"]),
                    "timestamp": datetime.now().isoformat()
                }, websocket)

            elif data.get("type") == "request_status":
                # Client explicitly requests current status
                status = await get_qa_status()
                summary = None
                if DB_AVAILABLE and qa_db:
                    try:
                        summary = qa_db.get_dashboard_summary()
                    except Exception:
                        pass
                await ws_manager.send_personal({
                    "type": "status_response",
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "summary": summary
                }, websocket)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# Helper function to broadcast events from other parts of the application
async def broadcast_qa_event(event_type: str, data: Dict[str, Any]):
    """
    Broadcast a QA event to all connected WebSocket clients.

    Args:
        event_type: Type of event (new_issue, issue_updated, run_started, run_completed)
        data: Event-specific data
    """
    await ws_manager.broadcast({
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    })


@router.get("/ws/status")
async def get_websocket_status():
    """Get WebSocket connection status."""
    return {
        "active_connections": len(ws_manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }
