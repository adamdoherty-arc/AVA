# New Feature Documentation Template

## Overview

This template ensures **uniform, comprehensive documentation** for all Magnus features. Every time a new feature is created or modified, follow this template to maintain consistency across the platform.

## Required Files (7 Total)

When creating a new feature, you MUST create all 7 documentation files:

### 1. README.md - User Guide
**Purpose**: User-facing documentation with step-by-step instructions

**Required Sections**:
```markdown
# [Feature Name] Feature

## Overview
[1-2 paragraph description of what this feature does]

## Key Capabilities
- Bullet list of main features
- What users can accomplish
- Primary use cases

## How to Use

### Step 1: [First Action]
[Detailed instructions with screenshots/descriptions]

### Step 2: [Second Action]
[Continue step-by-step]

## Understanding [Key Concept]
[Explain important concepts users need to know]

## Tips and Best Practices
- [Tip 1]
- [Tip 2]

## Troubleshooting

### Issue: [Common Problem]
**Solution**: [How to fix]

### Issue: [Another Problem]
**Solution**: [Resolution steps]

## FAQ

**Q: [Question]?**
A: [Answer]

## Related Features
- [Link to related feature 1]
- [Link to related feature 2]

## Additional Resources
- [External documentation]
- [Tutorial videos]
```

### 2. ARCHITECTURE.md - Technical Implementation
**Purpose**: Developer-focused technical details

**Required Sections**:
```markdown
# [Feature Name] - Architecture

## System Architecture

### Overview
[Architecture diagram description]

### Core Components
1. **Component 1** (`file_path.py`)
   - Responsibility
   - Key functions

2. **Component 2** (`file_path.py`)
   - Responsibility
   - Key functions

## Data Flow

```
User Action → Component A → Component B → Database → Response
```

## Key Algorithms

### Algorithm Name
```python
def algorithm_example():
    # Pseudo-code or actual implementation
    pass
```

## Dependencies

### External APIs
- API Name: Purpose, rate limits, authentication

### Database Tables
- Table Name: Columns, indexes, relationships

### Other Features
- Feature Name: What data/functionality is needed

## Performance Considerations
- Caching strategy
- Query optimization
- Rate limiting

## Error Handling
- Common errors and recovery
- Fallback mechanisms

## Security Considerations
- Authentication
- Data validation
- API key management

## Future Architecture Improvements
- [Planned improvement 1]
- [Planned improvement 2]
```

### 3. SPEC.md - Feature Specifications
**Purpose**: Detailed functional and technical requirements

**Required Sections**:
```markdown
# [Feature Name] - Specifications

## Functional Requirements

### FR-1: [Requirement Category]
#### FR-1.1: [Specific Requirement]
- **Description**: What it does
- **Priority**: High/Medium/Low
- **Status**: Implemented/Planned/In Progress

## UI Components

### Component Name
- **Location**: Where it appears
- **Elements**: Buttons, inputs, displays
- **Behavior**: User interactions
- **Validation**: Input rules

## Data Models

### Model Name
```python
{
    "field1": "type",
    "field2": "type",
    "description": "what this represents"
}
```

## Business Logic

### Logic Name
**Formula**: [Mathematical formula or algorithm]
**Thresholds**: [Key values and conditions]
**Example**: [Concrete example]

## API Specifications

### Endpoint Name
- **Method**: GET/POST/PUT/DELETE
- **URL**: /api/endpoint
- **Parameters**: List with types
- **Response**: Expected format
- **Errors**: Possible error codes

## Performance Requirements
- Response time: < X seconds
- Concurrent users: Y users
- Data volume: Z records

## Testing Specifications

### Unit Tests
- [Test case 1]
- [Test case 2]

### Integration Tests
- [Integration scenario 1]

### Edge Cases
- [Edge case 1]
- [Edge case 2]

## Success Metrics
- [Metric 1]: Target value
- [Metric 2]: Target value
```

### 4. WISHLIST.md - Future Enhancements
**Purpose**: Planned features and improvements

**Required Sections**:
```markdown
# [Feature Name] - Wishlist

## Priority 1 (Current Quarter)
### [Enhancement Name]
**Description**: What it does
**Value**: Why it's important
**Effort**: Time estimate
**Dependencies**: What's needed

## Priority 2 (Next Quarter)
[Same structure]

## Priority 3 (6-12 Months)
[Same structure]

## Priority 4 (Future/Long-term)
[Same structure]

## Community Requests
### [Request from users]
- **Requested by**: User/team
- **Use case**: Why they need it
- **Status**: Under review/Planned/Declined

## Technical Debt
- [Refactoring needed]
- [Performance optimization]

## Research Needed
- [Technology to investigate]
- [Proof of concept required]

## Implementation Roadmap

### Phase 1: [Name] (Timeline)
- Task 1
- Task 2

### Phase 2: [Name] (Timeline)
- Task 1
- Task 2
```

### 5. AGENT.md - Agent System Documentation
**Purpose**: AI agent capabilities and coordination

**Required Sections**:
```markdown
# [Feature Name] Feature Agent

## Agent Identity
- **Feature Name**: [Name]
- **Agent Version**: X.Y.Z
- **Feature Version**: X.Y.Z
- **Last Updated**: YYYY-MM-DD
- **Owner**: Magnus Platform
- **Status**: ✅ Active / 🚧 In Development / ⚠️ Deprecated

## Role & Responsibilities

### Primary Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]
3. [Responsibility 3]

### Data Sources
- **Source 1**: What data comes from where
- **Source 2**: Secondary data sources

## Feature Capabilities

### What This Agent CAN Do
- ✅ [Capability 1]
- ✅ [Capability 2]
- ✅ [Capability 3]

### What This Agent CANNOT Do
- ❌ [Limitation 1] (that's [Other Agent]'s role)
- ❌ [Limitation 2] (requires user action)
- ❌ [Limitation 3] (planned for future)

## Dependencies

### Required Features
- **Feature Name**: Why it's needed

### Optional Features
- **Feature Name**: Enhancement provided

### External APIs
- **API Name**: Purpose, authentication

### Database Tables
- **Table Name**: Usage

## Key Files & Code

### Main Implementation
- `file_path.py`: Lines X-Y (Description)
- `file_path.py`: Lines A-B (Description)

### Database Queries
```sql
-- Key query description
SELECT ...
```

## Current State

### Implemented Features
✅ [Feature 1]
✅ [Feature 2]

### Known Limitations
⚠️ [Limitation 1]
⚠️ [Limitation 2]

### Recent Changes
See [CHANGELOG.md](./CHANGELOG.md) for detailed history.

## Communication Patterns

### Incoming Requests

#### From Main Agent
```yaml
Request: "What this agent handles"
Response:
  - Data returned
  - Format used
```

#### From User
```yaml
Request: "User query pattern"
Response:
  - How agent responds
  - Data provided
```

### Outgoing Requests

#### To [Other Feature] Agent
```yaml
Request: "What data is needed"
Purpose: Why it's needed
Expected Response: Format expected
```

## Data Flow

```
User Action
    ↓
[Feature] Agent Receives
    ↓
Validate & Process
    ↓
Call Dependencies (if needed)
    ↓
Return Result
```

## Error Handling

### [Error Type]
```python
try:
    # Operation
except ErrorType as e:
    # Recovery strategy
```

## Performance Considerations

### Caching Strategy
- Cache what: Duration
- Invalidation: When

### Optimization
- [Optimization 1]
- [Optimization 2]

## Testing Checklist

### Before Deployment
- [ ] Feature loads without errors
- [ ] All data sources accessible
- [ ] Error handling works
- [ ] Performance acceptable

### Integration Tests
- [ ] Works with [Feature 1]
- [ ] Coordinates with [Feature 2]

## Maintenance

### When to Update This Agent

1. **User Request**: "Add X feature"
   - Update code
   - Update TODO.md
   - Document in CHANGELOG.md

2. **Bug Report**: "Y is broken"
   - Debug and fix
   - Test thoroughly
   - Document in CHANGELOG.md

3. **API Change**: External API updated
   - Update integration code
   - Test compatibility
   - Alert Main Agent

### Monitoring
- [Metric to track]: Threshold
- [Performance indicator]: Target

## Integration Points

### [API/Service Name]
- What data is exchanged
- How coordination happens

### Agent Coordination
- **Main Agent**: Overall coordination
- **[Feature] Agent**: Specific collaboration

## Future Enhancements

See [WISHLIST.md](./WISHLIST.md) for planned features.

## Questions This Agent Can Answer

1. "[Question 1]?"
2. "[Question 2]?"
...
10. "[Question 10]?"

## Questions This Agent CANNOT Answer

1. "[Question]" → [Other Agent] Agent
2. "[Question]" → Requires user input
...

---

**For detailed documentation, see:**
- [README.md](./README.md) - User guide
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical details
- [SPEC.md](./SPEC.md) - Specifications

**For current work, see:**
- [TODO.md](./TODO.md) - Active tasks
- [WISHLIST.md](./WISHLIST.md) - Future plans
- [CHANGELOG.md](./CHANGELOG.md) - Change history
```

### 6. TODO.md - Current Tasks
**Purpose**: Track active work, priorities, and known issues

**Required Sections**:
```markdown
# [Feature Name] - TODO List

## 🔴 High Priority (Current Sprint)

### [Task Name]
**Description**: What needs to be done
**Reason**: Why it's urgent
**Effort**: Time estimate
**Assigned**: Who's working on it
**Blockers**: What's preventing progress

## 🟡 Medium Priority (Next Sprint)

### [Task Name]
[Same structure as above]

## 🟢 Low Priority (Backlog)

### [Task Name]
[Same structure as above]

## 🐛 Known Issues

### [Bug Name]
**Description**: What's broken
**Impact**: High/Medium/Low
**Reproducible**: Yes/No
**Steps to Reproduce**:
1. Step 1
2. Step 2
**Workaround**: Temporary fix if available

## 📝 Technical Debt

### [Debt Item]
**Current State**: How it works now
**Desired State**: How it should work
**Risk**: What could go wrong
**Effort**: Time to fix

## 🧪 Testing Needed

### [Test Area]
**Coverage**: Current test coverage %
**Missing Tests**:
- [Test scenario 1]
- [Test scenario 2]

## 📚 Documentation

### [Doc Gap]
**Missing**: What's not documented
**Priority**: High/Medium/Low
**Owner**: Who should write it

## 👥 Community Requests

### [Request Name]
**Requested by**: User/team
**Description**: What they want
**Votes**: Number of upvotes
**Status**: Under review/Planned/Won't do

## 🎯 Implementation Roadmap

### This Week
- [ ] Task 1
- [ ] Task 2

### This Month
- [ ] Task 1
- [ ] Task 2

### This Quarter
- [ ] Task 1
- [ ] Task 2

## Last Updated
YYYY-MM-DD

---

**Related Documentation:**
- [WISHLIST.md](./WISHLIST.md) - Future enhancements
- [CHANGELOG.md](./CHANGELOG.md) - Completed changes
- [AGENT.md](./AGENT.md) - Agent capabilities
```

### 7. CHANGELOG.md - Version History
**Purpose**: Track all changes following Keep a Changelog format

**Required Sections**:
```markdown
# Changelog

All notable changes to the [Feature Name] feature will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- [Feature/enhancement being worked on]

### Changed
- [Modification to existing functionality]

### Fixed
- [Bug fix in progress]

### Removed
- [Deprecated feature being removed]

## [X.Y.Z] - YYYY-MM-DD

### Added
- **[Feature Name]**: Description of what was added
- **[Feature Name]**: Another addition

### Changed
- **[Component]**: What changed and why
- **[API]**: Breaking change details

### Fixed
- **[Bug]**: What was broken and how it's fixed
- **[Issue]**: Problem resolved

### Removed
- **[Feature]**: What was removed and why

### Security
- **[Vulnerability]**: Security fix details

## [X.Y.Z-1] - YYYY-MM-DD

[Continue with previous versions...]

## [1.0.0] - YYYY-MM-DD

### Added
- Initial release
- [Core feature 1]
- [Core feature 2]
- [Core feature 3]

---

## Version History

- **Major version** (X.0.0): Breaking changes, major features
- **Minor version** (X.Y.0): New features, backward compatible
- **Patch version** (X.Y.Z): Bug fixes, minor improvements

## Links

[Unreleased]: https://github.com/user/repo/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/user/repo/compare/vX.Y.Z-1...vX.Y.Z
[X.Y.Z-1]: https://github.com/user/repo/releases/tag/vX.Y.Z-1
```

## Feature Creation Checklist

When creating a new feature, complete this checklist:

### Phase 1: Planning
- [ ] Feature name chosen
- [ ] Purpose clearly defined
- [ ] User needs identified
- [ ] Dependencies mapped
- [ ] Reviewed with Main Agent

### Phase 2: Documentation
- [ ] Create feature folder: `features/[feature_name]/`
- [ ] Write README.md (user guide)
- [ ] Write ARCHITECTURE.md (technical)
- [ ] Write SPEC.md (requirements)
- [ ] Write WISHLIST.md (future plans)
- [ ] Write AGENT.md (agent system)
- [ ] Write TODO.md (current tasks)
- [ ] Write CHANGELOG.md (version 1.0.0)

### Phase 3: Integration
- [ ] Register in MAIN_AGENT.md
- [ ] Add to features/INDEX.md
- [ ] Update navigation in dashboard.py
- [ ] Document dependencies in other AGENT.md files
- [ ] Test integration with existing features

### Phase 4: Implementation
- [ ] Create feature code files
- [ ] Implement core functionality
- [ ] Add error handling
- [ ] Write unit tests
- [ ] Write integration tests

### Phase 5: Launch
- [ ] QA testing completed
- [ ] Documentation reviewed
- [ ] User guide validated
- [ ] Performance tested
- [ ] Security reviewed
- [ ] Deployed to production

## Maintenance Checklist

When updating an existing feature:

### Before Changes
- [ ] Read feature's AGENT.md
- [ ] Check TODO.md for priorities
- [ ] Review CHANGELOG.md for history
- [ ] Verify no conflicts with other features

### During Changes
- [ ] Update code
- [ ] Update tests
- [ ] Update documentation (if needed)

### After Changes
- [ ] Mark TODO item as complete
- [ ] Add entry to CHANGELOG.md
- [ ] Update AGENT.md (if capabilities changed)
- [ ] Update version numbers
- [ ] Notify Main Agent of cross-feature impacts

## File Naming Conventions

- **Folder name**: lowercase with underscores (e.g., `prediction_markets`)
- **README.md**: Exactly this name (uppercase)
- **ARCHITECTURE.md**: Exactly this name (uppercase)
- **SPEC.md**: Exactly this name (uppercase)
- **WISHLIST.md**: Exactly this name (uppercase)
- **AGENT.md**: Exactly this name (uppercase)
- **TODO.md**: Exactly this name (uppercase)
- **CHANGELOG.md**: Exactly this name (uppercase)

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
  - API changes that aren't backward compatible
  - Major feature overhauls
  - Database schema changes

- **MINOR** (X.Y.0): New features
  - New functionality added
  - Backward compatible changes
  - Enhancements to existing features

- **PATCH** (X.Y.Z): Bug fixes
  - Bug fixes
  - Performance improvements
  - Documentation updates

## Documentation Standards

### Writing Style
- **Clear and concise**: Short sentences, active voice
- **User-focused**: Write for the audience (users vs. developers)
- **Examples**: Include code examples and screenshots
- **Links**: Cross-reference related documentation

### Code Examples
```python
# Always include:
# 1. Context (what this does)
# 2. Working code (tested)
# 3. Comments explaining non-obvious parts

def example_function():
    """
    Brief description of what this does.

    Returns:
        What it returns
    """
    # Implementation
    return result
```

### Formatting
- Use markdown properly
- Include code fences with language
- Use tables for comparisons
- Use lists for sequences
- Use headings hierarchically

## Quality Standards

Every feature MUST have:

✅ **Complete Documentation** (all 7 files)
✅ **Working Code** (tested and functional)
✅ **Error Handling** (graceful failures)
✅ **Performance** (meets targets)
✅ **Security** (validated inputs, secure APIs)
✅ **Testing** (unit + integration tests)
✅ **Integration** (works with other features)

## Template Usage

### For New Features
1. Copy this template structure
2. Fill in all sections
3. Create all 7 files
4. Follow the checklist above

### For Existing Features
1. Compare with template
2. Fill in missing sections
3. Update to match format
4. Ensure consistency

## Example: Creating "Risk Analysis" Feature

```bash
# 1. Create folder
mkdir features/risk_analysis

# 2. Create all 7 files
touch features/risk_analysis/README.md
touch features/risk_analysis/ARCHITECTURE.md
touch features/risk_analysis/SPEC.md
touch features/risk_analysis/WISHLIST.md
touch features/risk_analysis/AGENT.md
touch features/risk_analysis/TODO.md
touch features/risk_analysis/CHANGELOG.md

# 3. Fill each file using templates above

# 4. Register with Main Agent
# Edit MAIN_AGENT.md to add risk_analysis

# 5. Update INDEX
# Edit features/INDEX.md to add new section
```

## Automation Tools (Future)

Planned automation:
- Script to generate all 7 files from template
- Auto-update TODO.md from git commits
- Auto-generate CHANGELOG.md from commits
- Validation script to check completeness

## Support

**Questions about this template?**
- See [MAIN_AGENT.md](MAIN_AGENT.md) for agent system
- See [features/INDEX.md](features/INDEX.md) for examples
- Check existing features for reference

**Template Updates:**
- This template will evolve based on feedback
- Check version in git history
- Suggest improvements via GitHub issues

---

**Last Updated**: 2025-11-01
**Template Version**: 1.0.0
**Maintained By**: Magnus Platform Team
