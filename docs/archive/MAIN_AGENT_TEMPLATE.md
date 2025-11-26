# Main Orchestrator Agent Template

## Purpose

This template is specifically for **Main/Orchestrator Agents** that coordinate multiple feature agents. This is **NOT** for feature agents (use FEATURE_TEMPLATE.md for those).

## When to Use This Template

Use this template when creating:
- ✅ Main orchestrator agents
- ✅ Coordination agents that manage multiple features
- ✅ Platform-level agents
- ❌ **NOT** for individual feature agents (use FEATURE_TEMPLATE.md)

---

## Main Orchestrator Agent Structure

### Required Sections

#### 1. Agent Identity & Metadata
```markdown
# [Agent Name] - Main Orchestrator Agent

## Agent Identity
- **Agent Type**: Main Orchestrator
- **Agent Version**: X.Y.Z
- **Platform Version**: X.Y.Z
- **Last Updated**: YYYY-MM-DD
- **Scope**: Platform-wide coordination
- **Status**: ✅ Active / 🚧 In Development
```

#### 2. Role & Purpose
```markdown
## Role & Responsibility

[Agent Name] is the **primary orchestrator** for [Platform Name]. It serves as the central intelligence that:

1. **Routes requests** to appropriate feature agents
2. **Coordinates multi-feature workflows** requiring collaboration
3. **Maintains platform-wide context** and state
4. **Ensures consistency** across all features
5. **Monitors system health** and dependencies
```

#### 3. Architecture Overview
```markdown
## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          MAIN ORCHESTRATOR AGENT                │
│  (Routes, Coordinates, Maintains Context)       │
└──────────────────┬──────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Feature  │ │ Feature  │ │ Feature  │
│ Agent 1  │ │ Agent 2  │ │ Agent N  │
└──────────┘ └──────────┘ └──────────┘
```
```

#### 4. Feature Agent Registry
```markdown
## Feature Agents Registry

The Main Agent can delegate to the following specialized feature agents:

| Feature | Agent File | Responsible For |
|---------|-----------|-----------------|
| [Feature 1] | `features/[name]/AGENT.md` | [Description] |
| [Feature 2] | `features/[name]/AGENT.md` | [Description] |
| [Feature N] | `features/[name]/AGENT.md` | [Description] |
```

#### 5. Communication Protocol
```markdown
## Communication Protocol

### Request Routing

When a user makes a request, the Main Agent:

1. **Analyzes the request** to determine which feature(s) are involved
2. **Consults the appropriate AGENT.md** file(s)
3. **Delegates to feature agent(s)** with full context
4. **Aggregates responses** if multiple agents involved
5. **Returns coordinated result** to user

### Inter-Agent Communication

Feature agents can request assistance from other feature agents through the Main Agent:

```
User Request → Main Agent → Feature Agent A
                   ↓
        Feature Agent A needs data from Agent B
                   ↓
        Main Agent → Feature Agent B → Returns data
                   ↓
        Main Agent → Feature Agent A (with B's data)
                   ↓
        Main Agent → User (final response)
```
```

#### 6. Routing Decision Matrix
```markdown
## Routing Decision Matrix

| User Request Pattern | Primary Agent | Supporting Agents |
|---------------------|---------------|-------------------|
| "[Request 1]" | [Agent 1] | [Agent 2, Agent 3] |
| "[Request 2]" | [Agent 2] | [Agent 1] |
| "[Request N]" | [Agent N] | - |
```

#### 7. Context Management
```markdown
## Context Management

The Main Agent maintains platform-wide context including:

### Session State
- [Context item 1]
- [Context item 2]

### Cross-Feature Data
- [Shared data 1]
- [Shared data 2]

### Feature Health Status
- Last sync times for each feature
- Error states and recovery status
- Cache freshness indicators
```

#### 8. Multi-Feature Workflows
```markdown
## Multi-Feature Workflows

### Example: [Workflow Name]

```yaml
Request: "[User request]"

Workflow:
  1. Main Agent receives request
  2. Delegates to [Feature Agent 1] → [Action]
  3. Delegates to [Feature Agent 2] → [Action]
  4. Main Agent aggregates data
  5. Main Agent coordinates recommendation
  6. Returns comprehensive result
```
```

#### 9. Change Tracking & Coordination
```markdown
## Change Tracking & History

The Main Agent monitors all feature changes through their CHANGELOG.md files:

### On Feature Update:
1. Feature agent updates its `CHANGELOG.md`
2. Feature agent updates its `TODO.md` (removes completed items)
3. Main Agent detects changes
4. Main Agent updates dependent features if needed
5. Main Agent logs cross-feature impacts

### Version Compatibility:
- Each feature maintains its version in `AGENT.md`
- Main Agent ensures compatibility between features
- Alerts on breaking changes between features
```

#### 10. Knowledge Base
```markdown
## Knowledge Base

The Main Agent has deep knowledge of:

### Platform Architecture
- [Architecture component 1]
- [Architecture component 2]

### Domain Knowledge
- [Domain knowledge 1]
- [Domain knowledge 2]

### Feature Dependencies
```
[Feature 1] → depends on → [[Feature 2], [Feature 3]]
[Feature 2] → depends on → [[External API], [Database]]
```
```

#### 11. Agent Capabilities
```markdown
## Agent Capabilities

### What the Main Agent CAN do:
- ✅ [Capability 1]
- ✅ [Capability 2]
- ✅ [Capability N]

### What the Main Agent CANNOT do:
- ❌ [Limitation 1] (delegates to [Feature Agent])
- ❌ [Limitation 2] (requires user confirmation)
- ❌ [Limitation N]
```

#### 12. Emergency Protocols
```markdown
## Emergency Protocols

### Feature Failure
If a feature agent fails:
1. Main Agent isolates the failure
2. Continues operation with remaining features
3. Provides degraded functionality warnings
4. Logs detailed error for debugging

### Data Inconsistency
If cross-feature data conflicts:
1. Main Agent identifies conflict source
2. Determines authoritative data source
3. Coordinates data reconciliation
4. Updates all dependent features

### API Outage
If external API fails:
1. Main Agent switches to cached data
2. Updates UI with "stale data" warnings
3. Queues retry attempts
4. Notifies user of degraded service
```

#### 13. Performance Monitoring
```markdown
## Performance Monitoring

The Main Agent tracks:

- **Response times** for each feature agent
- **Cache hit rates** across features
- **API call frequencies** and rate limits
- **Database query performance**
- **User request patterns**

### Optimization Triggers
- If feature response > [threshold] → investigate bottleneck
- If cache miss rate > [threshold] → adjust caching strategy
- If API rate limit reached → implement throttling
```

#### 14. User Interaction Patterns
```markdown
## User Interaction Patterns

### Question Answering
```
User: "[Question]"
Main Agent → [Feature Agent] → Returns [Answer]
```

### Feature Requests
```
User: "[Request]"
Main Agent → [Agent(s)] → Coordinate implementation
```

### Troubleshooting
```
User: "[Problem]"
Main Agent → [Feature Agent] → Diagnose and report issue
```

### Multi-Step Tasks
```
User: "[Complex task]"
Main Agent:
  1. [Step 1]
  2. [Step 2]
  3. [Step N]
```
```

#### 15. Documentation Maintenance
```markdown
## Documentation Maintenance

The Main Agent ensures:

1. All feature `AGENT.md` files follow consistent format
2. All `TODO.md` files are current and prioritized
3. All `CHANGELOG.md` files track significant changes
4. Cross-feature impacts are documented
5. Main feature index is updated with new features
```

#### 16. Testing & Quality Assurance
```markdown
## Testing & Quality Assurance

### Pre-Change Testing Checklist
- [ ] All imports resolve correctly
- [ ] All feature agents accessible
- [ ] No circular dependencies
- [ ] API connections valid
- [ ] Database accessible
- [ ] Cache layer operational

### Post-Change Testing Checklist
- [ ] Platform loads without errors
- [ ] All feature pages accessible
- [ ] Navigation works correctly
- [ ] No console errors
- [ ] Performance acceptable
- [ ] Documentation updated

### Automated Tests
- Unit tests: `pytest tests/test_main_agent.py`
- Integration tests: `pytest tests/test_agent_coordination.py`
- End-to-end tests: `pytest tests/test_workflows.py`

### Manual QA Checklist
- [ ] Test routing to each feature
- [ ] Test multi-feature workflows
- [ ] Test error handling
- [ ] Test degraded mode
- [ ] Verify documentation accuracy
```

#### 17. Version Information
```markdown
## Version Information

- **Agent Version**: X.Y.Z
- **Platform Version**: X.Y.Z
- **Last Updated**: YYYY-MM-DD
- **Supported Features**: N
- **Active Integrations**: [API 1], [API 2], [Database], [Cache]
- **Documentation Template**: MAIN_AGENT_TEMPLATE.md vX.Y.Z
- **Interaction Protocol**: [Protocol document] vX.Y.Z
```

#### 18. Maintenance Schedule
```markdown
## Maintenance Schedule

### Daily Tasks
- Monitor feature health
- Check API rate limits
- Verify data consistency
- Update cache freshness

### Weekly Tasks
- Review feature TODOs
- Coordinate feature updates
- Check for breaking changes
- Update documentation

### Monthly Tasks
- Performance audit
- Security review
- Dependency updates
- User feedback analysis
```

---

## Quality Standards for Main Agents

Every Main Orchestrator Agent MUST have:

✅ **Clear Scope** - Platform-wide coordination defined
✅ **Feature Registry** - All managed features listed
✅ **Routing Logic** - Request routing clearly documented
✅ **Communication Protocol** - Inter-agent communication specified
✅ **Error Handling** - Emergency protocols documented
✅ **Performance Monitoring** - Metrics and thresholds defined
✅ **Testing Checklist** - Pre/post-change QA steps
✅ **Version Tracking** - Compatible feature versions tracked

---

## Differences from Feature Template

| Aspect | Main Agent Template | Feature Template |
|--------|-------------------|------------------|
| **Scope** | Platform-wide | Single feature |
| **Coordination** | Manages many agents | Works alone or with Main |
| **Routing** | Routes all requests | Handles own requests |
| **Context** | Platform-wide | Feature-specific |
| **Documentation** | Single MAIN_AGENT.md | 7 files (README, ARCH, SPEC, etc.) |
| **Dependencies** | Knows all features | Knows only its dependencies |
| **Error Handling** | Graceful degradation | Feature-specific errors |
| **Testing** | Integration + E2E | Unit + integration |

---

## Example Usage

### Creating a New Main Agent

```bash
# 1. Copy this template
cp MAIN_AGENT_TEMPLATE.md MY_MAIN_AGENT.md

# 2. Fill in all required sections
# - Update agent identity
# - List all managed features
# - Define routing logic
# - Document communication protocol
# - Add emergency protocols
# - Create testing checklist

# 3. Register with platform
# - Add to main documentation
# - Update platform architecture docs
# - Inform all feature agents

# 4. Test thoroughly
pytest tests/test_my_main_agent.py
pytest tests/test_integration.py
pytest tests/test_e2e.py
```

### When NOT to Use This Template

❌ **Don't use for**:
- Individual features (use FEATURE_TEMPLATE.md)
- Sub-components of features
- Helper utilities
- Data models

✅ **Do use for**:
- Platform orchestrators
- Multi-feature coordinators
- System-level agents

---

## Automated Validation

Before considering a Main Agent complete, run:

```bash
# Check template compliance
python scripts/validate_main_agent.py MY_MAIN_AGENT.md

# Expected output:
# ✅ All required sections present
# ✅ Feature registry complete
# ✅ Routing logic documented
# ✅ Testing checklist included
# ✅ Version information current
```

---

## Template Maintenance

This template is maintained by the platform team.

**Current Version**: 1.0.0
**Last Updated**: 2025-11-01
**Changelog**: See MAIN_AGENT_TEMPLATE_CHANGELOG.md

**To suggest improvements**:
1. Create issue: "MAIN_AGENT_TEMPLATE: [suggestion]"
2. Discuss with platform team
3. Submit PR with changes
4. Update version number

---

**Remember**: Main Agents are orchestrators, not features. They coordinate, route, and maintain consistency across the platform.

**For feature agents, use**: [FEATURE_TEMPLATE.md](FEATURE_TEMPLATE.md)
