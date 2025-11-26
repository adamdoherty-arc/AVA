# Master Agent System Architecture Plan

**Created**: 2025-10-27
**Status**: Planning Phase
**Purpose**: Transform the Wheel Strategy project into an evergrowing, agent-orchestrated system

---

## Executive Summary

This document outlines the comprehensive plan to create a **Master Agent System** that will:
1. Orchestrate all existing and future agents
2. Provide intelligent routing of tasks to appropriate specialist agents
3. Enable the system to grow organically with new agents
4. Automate testing, code review, and deployment workflows
5. Maintain project health and code quality autonomously

---

## Current State Analysis

### Existing Agent Files

**Location**: `src/agents/`

1. **wheel_strategy_agent.py** (396 lines)
   - Finds optimal put and call opportunities
   - Implements wheel strategy logic
   - Calculates Greeks using Black-Scholes
   - Uses yfinance for data

2. **risk_management_agent.py** (457 lines)
   - Validates trades against risk parameters
   - Calculates position sizing
   - Analyzes portfolio risk
   - Manages sector allocation

3. **alert_agent.py** (406 lines)
   - Sends notifications through multiple channels
   - Monitors alerts and conditions
   - Formats alert messages
   - Supports email, Discord, Telegram

4. **market_data_agent.py** (272 lines)
   - Scans for opportunities
   - Monitors real-time prices
   - Integrates TradingView signals
   - Filters stocks by criteria

### Existing Claude Agent Documentation

**Location**: `.claude/`

1. **AGENT_WORKFLOW_SPEC.md** - 11 specialist agents defined
2. **AGENT_USAGE_GUIDE.md** - Practical usage patterns
3. **settings.local.json** - Claude Code configuration

### Key Project Components

**Dashboards**:
- `dashboard.py` - Main Streamlit dashboard with all pages
- `premium_hunter.py` - Simplified premium opportunity finder

**Data Sync Services**:
- `src/watchlist_sync_service.py` - Background watchlist syncing
- `src/enhanced_options_fetcher.py` - Multi-expiration delta targeting
- `src/tradingview_api_sync.py` - TradingView API integration
- `src/stock_data_sync.py` - Stock price updates

**Database**:
- PostgreSQL "Magnus" database
- Tables: `tv_watchlists_api`, `tv_symbols_api`, `stock_data`, `stock_premiums`

---

## Architecture Design

### Master Agent Hierarchy

```
                    ┌─────────────────────┐
                    │   MASTER AGENT      │
                    │   (Orchestrator)    │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐          ┌──────▼──────┐       ┌──────▼──────┐
   │ CLAUDE  │          │   PYTHON    │       │  PROJECT    │
   │ AGENTS  │          │   RUNTIME   │       │  MGMT       │
   │ LAYER   │          │   AGENTS    │       │  AGENTS     │
   └────┬────┘          └──────┬──────┘       └──────┬──────┘
        │                      │                      │
┌───────┴────────┐    ┌────────┴────────┐    ┌───────┴────────┐
│ • QA/Debugger  │    │ • Wheel Strategy│    │ • Planner      │
│ • Code Reviewer│    │ • Risk Mgmt     │    │ • Innovation   │
│ • Architect    │    │ • Market Data   │    │ • Documentation│
│ • Python Pro   │    │ • Alert Agent   │    │ • Monitoring   │
│ • DB Optimizer │    │                 │    │                │
│ • Performance  │    │                 │    │                │
└────────────────┘    └─────────────────┘    └────────────────┘
```

### Agent Categories

#### 1. Claude Code Agents (Task Tool Based)
**Location**: Invoked via Claude Code Task tool
**Purpose**: Development, testing, review, architecture

- **QA/Debugger**: Tests code before delivery (MANDATORY)
- **Code Reviewer**: Reviews for quality, security, best practices
- **Architect**: Designs system architecture and features
- **Python Pro**: Writes idiomatic Python code
- **Database Optimizer**: Optimizes queries and schema
- **Performance Engineer**: Profiles and optimizes bottlenecks
- **Frontend Developer**: Builds Streamlit UI components
- **Error Detective**: Root cause analysis for bugs
- **Innovation Agent**: Recommends future improvements
- **Documentation Agent**: Maintains comprehensive docs

#### 2. Python Runtime Agents (Asyncio Based)
**Location**: `src/agents/`
**Purpose**: Real-time trading operations

- **WheelStrategyAgent**: Find opportunities, calculate Greeks
- **RiskManagementAgent**: Validate trades, manage risk
- **MarketDataAgent**: Monitor prices, scan opportunities
- **AlertAgent**: Send notifications and alerts

#### 3. Project Management Agents (Master Orchestration)
**Location**: `src/agents/master/`
**Purpose**: Coordinate workflows, track tasks, manage state

- **MasterAgent**: Main orchestrator
- **TaskRouter**: Routes requests to appropriate agents
- **StateManager**: Maintains project and agent state
- **AgentRegistry**: Tracks available agents and capabilities

---

## Proposed Folder Structure

```
c:\Code\WheelStrategy\
├── .claude/
│   ├── AGENT_WORKFLOW_SPEC.md
│   ├── AGENT_USAGE_GUIDE.md
│   ├── MASTER_AGENT_PLAN.md (NEW)
│   └── settings.local.json
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   │
│   │   ├── master/                        (NEW)
│   │   │   ├── __init__.py
│   │   │   ├── master_agent.py           (Orchestrator)
│   │   │   ├── task_router.py            (Route to specialists)
│   │   │   ├── agent_registry.py         (Track all agents)
│   │   │   ├── state_manager.py          (Maintain state)
│   │   │   └── workflow_engine.py        (Execute workflows)
│   │   │
│   │   ├── runtime/                       (REORGANIZED)
│   │   │   ├── __init__.py
│   │   │   ├── wheel_strategy_agent.py   (MOVED)
│   │   │   ├── risk_management_agent.py  (MOVED)
│   │   │   ├── market_data_agent.py      (MOVED)
│   │   │   └── alert_agent.py            (MOVED)
│   │   │
│   │   ├── claude/                        (NEW - References)
│   │   │   ├── __init__.py
│   │   │   ├── qa_agent_spec.py          (QA agent metadata)
│   │   │   ├── code_review_spec.py       (Code review metadata)
│   │   │   └── agent_specs.py            (All Claude agent specs)
│   │   │
│   │   └── project/                       (NEW)
│   │       ├── __init__.py
│   │       ├── planner_agent.py          (Task planning)
│   │       ├── monitor_agent.py          (System health)
│   │       └── innovation_agent.py       (Suggest improvements)
│   │
│   ├── enhanced_options_fetcher.py
│   ├── watchlist_sync_service.py
│   ├── tradingview_api_sync.py
│   └── ... (other existing files)
│
├── dashboard.py
├── premium_hunter.py
├── CURRENT_STATUS.md
└── MASTER_AGENT_PLAN.md (THIS FILE)
```

---

## Master Agent System Components

### 1. Master Agent (`master_agent.py`)

**Core Responsibilities**:
- Receive user requests or system events
- Analyze request type and complexity
- Route to appropriate agent(s)
- Coordinate multi-agent workflows
- Track task progress
- Ensure quality gates are met
- Aggregate results and report back

**Key Methods**:
```python
class MasterAgent:
    async def handle_request(self, request: Request) -> Response
    async def route_task(self, task: Task) -> Agent
    async def execute_workflow(self, workflow: Workflow) -> Result
    async def coordinate_agents(self, agents: List[Agent]) -> Result
    async def ensure_quality(self, code_changes: CodeChanges) -> QualityReport
    async def learn_from_execution(self, execution: Execution) -> Insights
```

### 2. Agent Registry (`agent_registry.py`)

**Purpose**: Central registry of all available agents

**Registry Structure**:
```python
AGENT_REGISTRY = {
    # Claude Code Agents (invoked via Task tool)
    "qa_debugger": {
        "type": "claude",
        "subagent_type": "debugger",
        "capabilities": ["test", "debug", "verify"],
        "trigger_conditions": ["code_change", "before_delivery"],
        "mandatory": True,
        "priority": "critical"
    },
    "code_reviewer": {
        "type": "claude",
        "subagent_type": "code-reviewer",
        "capabilities": ["review", "security", "quality"],
        "trigger_conditions": ["major_change", "before_merge"],
        "priority": "high"
    },

    # Python Runtime Agents
    "wheel_strategy": {
        "type": "runtime",
        "class": "WheelStrategyAgent",
        "capabilities": ["find_opportunities", "calculate_greeks"],
        "trigger_conditions": ["scan_request", "opportunity_search"]
    },

    # ... all other agents
}
```

### 3. Task Router (`task_router.py`)

**Purpose**: Intelligent routing of tasks to appropriate agents

**Routing Logic**:
```python
class TaskRouter:
    async def route(self, task: Task) -> List[Agent]:
        """
        Analyze task and return appropriate agents.

        Examples:
        - "Fix bug" → [Debugger, Python Pro, QA]
        - "New feature" → [Architect, Planner, Python Pro, Frontend, QA, Code Reviewer]
        - "Optimize query" → [Database Optimizer, QA]
        - "Find opportunities" → [Wheel Strategy, Risk Management, Alert]
        """

    def analyze_task_type(self, task: Task) -> TaskType
    def select_agents(self, task_type: TaskType) -> List[AgentSpec]
    def order_execution(self, agents: List[AgentSpec]) -> ExecutionPlan
```

### 4. Workflow Engine (`workflow_engine.py`)

**Purpose**: Execute multi-agent workflows

**Standard Workflows**:

**Bug Fix Workflow**:
```
1. Debugger Agent → Identify root cause
2. Python Pro → Implement fix
3. QA Agent → Test fix
4. Code Reviewer → Review quality
5. Documentation → Update changelog
```

**New Feature Workflow**:
```
1. Architect → Design solution
2. Planner → Create tasks with todos
3. Python Pro + Frontend + DB Optimizer → Implement (parallel)
4. QA Agent → Comprehensive testing
5. Code Reviewer → Quality check
6. Documentation → Write guide
7. Innovation → Suggest related improvements
```

**Performance Issue Workflow**:
```
1. Performance Engineer → Profile and identify bottleneck
2. Database Optimizer → Optimize queries (if DB issue)
3. Python Pro → Implement optimizations
4. QA Agent → Verify performance improvement
5. Monitor Agent → Track metrics
```

### 5. State Manager (`state_manager.py`)

**Purpose**: Maintain project and agent state

**State Tracking**:
- Active tasks and their status
- Agent execution history
- Quality metrics
- System health
- Recent changes
- Test results
- Performance metrics

**Persistence**: Redis or PostgreSQL

---

## Integration with Existing Systems

### Claude Code Agents Integration

**How it works**:
1. Master Agent receives request
2. Determines Claude agent needed
3. Invokes using Task tool with detailed prompt
4. Waits for agent completion
5. Processes agent output
6. Routes to next agent if needed

**Example**:
```python
# Master Agent decides to use QA agent
await self.invoke_claude_agent(
    subagent_type="debugger",
    prompt=f"""
    Test the following changes:
    - Files modified: {modified_files}
    - Features added: {features}
    - Expected behavior: {expected}

    Run these checks:
    1. Syntax validation
    2. Import checking
    3. Dashboard loads without errors
    4. All new features work correctly
    5. No breaking changes to existing functionality

    Report detailed test results.
    """
)
```

### Python Runtime Agents Integration

**How it works**:
1. Master Agent imports and instantiates runtime agents
2. Calls async methods directly
3. Agents perform real-time operations
4. Results aggregated by Master Agent

**Example**:
```python
# Master Agent coordinates trading agents
wheel_agent = WheelStrategyAgent(redis_client)
risk_agent = RiskManagementAgent(redis_client)

opportunities = await wheel_agent.find_put_opportunities(symbols)
for opp in opportunities:
    validation = await risk_agent.validate_trade(opp, portfolio)
    if validation['valid']:
        await alert_agent.send_opportunity_alert(opp)
```

---

## Agent Communication Protocol

### Request Format
```python
{
    "request_id": "uuid",
    "type": "task|query|command",
    "priority": "critical|high|medium|low",
    "requester": "user|system|agent",
    "task": {
        "action": "fix_bug|add_feature|optimize|test",
        "target": "file_path or component",
        "details": {},
        "context": {}
    },
    "timestamp": "iso8601"
}
```

### Response Format
```python
{
    "request_id": "uuid",
    "agent": "agent_name",
    "status": "success|failure|partial",
    "result": {},
    "errors": [],
    "warnings": [],
    "next_steps": [],
    "timestamp": "iso8601"
}
```

### Agent-to-Agent Communication
```python
{
    "from_agent": "agent_name",
    "to_agent": "agent_name",
    "message_type": "handoff|request|notification",
    "context": {},
    "deliverables": {},
    "action_required": "",
    "priority": "high|medium|low"
}
```

---

## Quality Gates & Testing Protocol

### Mandatory Quality Gates

**Gate 1: Syntax & Imports** (Automated)
- Code must parse without errors
- All imports must resolve
- No obvious syntax issues
- **Agent**: Python syntax checker

**Gate 2: Unit Tests** (QA Agent)
- All new code has tests
- Existing tests still pass
- Edge cases covered
- **Agent**: QA/Debugger Agent

**Gate 3: Integration Tests** (QA Agent)
- Feature works end-to-end
- No breaking changes
- Dashboard loads correctly
- Database queries work
- **Agent**: QA/Debugger Agent

**Gate 4: Code Quality** (Code Reviewer)
- Follows Python best practices (PEP 8)
- Properly documented
- Error handling present
- No security issues
- **Agent**: Code Reviewer Agent

**Gate 5: Performance** (Performance Engineer)
- Queries < 1 second
- Page load < 3 seconds
- Memory usage reasonable
- No resource leaks
- **Agent**: Performance Engineer Agent (for critical paths)

### Testing Automation

**Before EVERY delivery**:
```python
async def pre_delivery_checks(changes: CodeChanges) -> QualityReport:
    """
    MANDATORY checks before user sees changes.
    Master Agent orchestrates this workflow.
    """

    # Gate 1: Syntax
    syntax_ok = await check_syntax(changes.files)
    if not syntax_ok:
        return QualityReport(passed=False, gate="syntax")

    # Gate 2: QA Testing
    qa_result = await invoke_qa_agent(changes)
    if not qa_result.passed:
        return QualityReport(passed=False, gate="qa", details=qa_result)

    # Gate 3: Code Review (for major changes)
    if changes.is_major:
        review_result = await invoke_code_reviewer(changes)
        if review_result.score < 90:
            return QualityReport(passed=False, gate="review", details=review_result)

    return QualityReport(passed=True, all_gates_passed=True)
```

---

## Autonomous Operations

### Self-Healing Capabilities

**1. Error Detection**:
- Monitor Agent detects errors in logs
- Master Agent receives alert
- Error Detective Agent analyzes root cause
- Python Pro Agent implements fix
- QA Agent tests fix
- Auto-deploy if all tests pass

**2. Performance Monitoring**:
- Monitor Agent tracks query times
- If degradation detected → Performance Engineer Agent
- Optimization implemented
- QA Agent verifies improvement

**3. Dependency Updates**:
- Monitor Agent checks for security updates
- Master Agent creates update plan
- Python Pro updates dependencies
- QA Agent runs full test suite
- Deploy if tests pass

### Scheduled Operations

**Daily**:
- Health check of all systems
- Update stock prices and premiums
- Scan for new opportunities
- Check for agent improvements

**Weekly**:
- Comprehensive code review
- Performance analysis
- Innovation agent recommendations
- Documentation updates

**Monthly**:
- Full system audit
- Architecture review
- Dependency upgrades
- Security scan

---

## Learning & Improvement

### Execution Tracking

Track every agent execution:
- What task was performed
- Which agents were used
- Time taken
- Success/failure
- Errors encountered
- User satisfaction (if available)

### Pattern Recognition

Identify patterns:
- Common error types
- Frequently used agent combinations
- Time-consuming workflows
- High-success approaches

### Optimization

Use patterns to:
- Improve routing decisions
- Pre-fetch likely needed agents
- Optimize workflow sequences
- Suggest proactive improvements

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Create folder structure
- [ ] Implement Agent Registry
- [ ] Implement Task Router (basic)
- [ ] Create Master Agent skeleton
- [ ] Test with simple workflows

### Phase 2: Core Integration (Week 2)
- [ ] Integrate Claude Code agents
- [ ] Integrate Python runtime agents
- [ ] Implement State Manager
- [ ] Build Workflow Engine
- [ ] Test multi-agent workflows

### Phase 3: Automation (Week 3)
- [ ] Implement quality gates
- [ ] Build automated testing workflows
- [ ] Add pre-delivery checks
- [ ] Create monitoring capabilities
- [ ] Test autonomous operations

### Phase 4: Intelligence (Week 4)
- [ ] Add pattern recognition
- [ ] Implement learning from history
- [ ] Build recommendation engine
- [ ] Optimize routing decisions
- [ ] Create self-improvement loops

### Phase 5: Polish & Scale (Week 5+)
- [ ] Comprehensive documentation
- [ ] Performance optimization
- [ ] Error handling refinement
- [ ] Add new specialized agents as needed
- [ ] Continuous improvement

---

## Success Metrics

### System Health
- All tests pass before delivery: **100%**
- Code review score: **> 90%**
- Zero user-reported bugs from tested code: **Target**
- Mean time to fix issues: **< 1 hour**

### Agent Performance
- Task routing accuracy: **> 95%**
- Workflow completion rate: **> 98%**
- Agent availability: **99.9%**
- Average task completion time: **Track and optimize**

### Code Quality
- Test coverage: **> 80%**
- No critical security issues: **100%**
- Performance benchmarks met: **> 95%**
- Documentation completeness: **> 90%**

---

## Risk Mitigation

### Agent Failure Handling
- **Problem**: Agent fails or times out
- **Solution**: Fallback to simpler agent or manual intervention
- **Prevention**: Health checks, timeouts, retry logic

### Infinite Loops
- **Problem**: Agents keep calling each other
- **Solution**: Max recursion depth, execution timeout
- **Prevention**: Clear workflow termination conditions

### State Corruption
- **Problem**: State manager data becomes inconsistent
- **Solution**: State validation, rollback capabilities
- **Prevention**: Atomic operations, versioning

### Quality Gate Bypass
- **Problem**: Pressure to skip testing
- **Solution**: MANDATORY gates, no bypass except documented emergencies
- **Prevention**: Fast test execution, parallel testing

---

## Future Enhancements

### Agent Marketplace
- Community-contributed agents
- Plugin architecture
- Agent versioning
- Dependency management

### Natural Language Interface
- "Find me high-premium opportunities under $30"
- "Optimize the slow query in dashboard.py"
- "Run full test suite and fix any issues"

### Predictive Capabilities
- Predict which agents will be needed
- Suggest improvements before issues occur
- Proactive optimization
- Trend analysis

### Multi-Project Support
- Master Agent manages multiple projects
- Shared agent pool
- Cross-project learning
- Resource optimization

---

## Conclusion

This Master Agent System will transform the Wheel Strategy project from a manual development process into an **autonomous, self-improving system** that:

1. ✅ **Never delivers untested code** (QA Agent is mandatory)
2. ✅ **Maintains high code quality** (Code Reviewer checks all major changes)
3. ✅ **Continuously improves** (Innovation Agent suggests enhancements)
4. ✅ **Scales organically** (New agents easily added via registry)
5. ✅ **Learns from experience** (Pattern recognition and optimization)

The system embodies the principles learned from past iterations:
- **ALWAYS test with debugger agent first**
- **Keep UI simple and clear**
- **Let agents find bugs, not users**
- **Proactive quality over reactive fixes**

---

**Next Steps**:
1. Review and approve this plan
2. Begin Phase 1 implementation
3. Create master agent system
4. Test with existing workflows
5. Iterate and improve

**Status**: ✅ PLAN COMPLETE - Ready for Implementation
