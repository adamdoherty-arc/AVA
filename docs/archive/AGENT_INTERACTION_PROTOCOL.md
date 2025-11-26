# Magnus Agent Interaction Protocol

## Purpose

This document defines how to interact with the Magnus Main Agent and feature agents. It ensures consistency, proper documentation updates, and adherence to the **FEATURE_TEMPLATE.md** standard.

## 🔔 IMPORTANT REMINDER

Every time you interact with the Main Agent or any feature agent, you will be reminded to:

1. **Follow FEATURE_TEMPLATE.md** for all documentation
2. **Update TODO.md** when tasks are started or completed
3. **Update CHANGELOG.md** when changes are made
4. **Update WISHLIST.md** when new ideas arise
5. **Update ARCHITECTURE.md** when structure changes
6. **Maintain uniformity** across all features

## Main Agent Reminder System

### Every Interaction Starts With:

```
🤖 MAGNUS MAIN AGENT
📋 Template: FEATURE_TEMPLATE.md
📍 Feature: [Current Feature Name]

Before proceeding, ensure:
✓ FEATURE_TEMPLATE.md is being followed
✓ All 7 documentation files exist
✓ Updates will be reflected in TODO, CHANGELOG, WISHLIST

How can I help you today?
```

## Feature Agent Reminder System

### Every Feature-Specific Interaction Includes:

```
🤖 [FEATURE NAME] AGENT
📋 Template: FEATURE_TEMPLATE.md
📂 Location: features/[feature_name]/

Documentation Status:
✓ README.md - User Guide
✓ ARCHITECTURE.md - Technical Details
✓ SPEC.md - Requirements
✓ WISHLIST.md - Future Plans
✓ AGENT.md - Agent System
✓ TODO.md - Current Tasks
✓ CHANGELOG.md - Version History

Ready to assist with [Feature Name].
Remember to update documentation after changes!
```

## Interaction Workflow

### 1. User Asks Question/Request

Example: "Add earnings warnings to the Dashboard"

### 2. Agent Identifies Feature(s)

Main Agent determines:
- Primary feature: Dashboard
- Supporting features: Earnings Calendar
- Coordination needed: Yes

### 3. Agent Reminder Displayed

```
🤖 MAGNUS MAIN AGENT
📋 Template: FEATURE_TEMPLATE.md
📍 Features Involved: Dashboard, Earnings Calendar

This request will require:
1. Dashboard Agent - Main implementation
2. Earnings Calendar Agent - Data source
3. Updates to both feature's documentation

Documentation updates needed:
- features/dashboard/TODO.md (add task)
- features/dashboard/CHANGELOG.md (after completion)
- features/dashboard/ARCHITECTURE.md (integration section)
- features/earnings_calendar/AGENT.md (add caller)

Proceeding with analysis...
```

### 4. Agent Provides Response

Main Agent coordinates with feature agents and provides comprehensive answer.

### 5. Documentation Update Reminder

After completing work:

```
✅ Task Completed!

📝 REQUIRED DOCUMENTATION UPDATES:

features/dashboard/TODO.md:
  - [x] Add earnings warnings integration

features/dashboard/CHANGELOG.md:
  - Add to [Unreleased] → Added section:
    "Earnings warnings integration with calendar"

features/dashboard/ARCHITECTURE.md:
  - Update Dependencies section
  - Add Earnings Calendar integration

features/earnings_calendar/AGENT.md:
  - Update "Questions This Agent Can Answer"
  - Add Dashboard to integration points

Remember: All updates must follow FEATURE_TEMPLATE.md format!
```

## Response Format Standards

### Question Answering

```markdown
🤖 [AGENT NAME]
📋 Template: FEATURE_TEMPLATE.md

**Question**: [User's question]

**Answer**: [Comprehensive response]

**Related Documentation**:
- [Link to relevant AGENT.md section]
- [Link to README.md section]

**Documentation Status**: ✅ Up to date / ⚠️ Needs update
```

### Feature Modification

```markdown
🤖 MAGNUS MAIN AGENT
📋 Template: FEATURE_TEMPLATE.md
📍 Feature: [Feature Name]

**Request**: [What user wants]

**Impact Analysis**:
- Affects: [List of features]
- Requires: [Dependencies]
- Effort: [Time estimate]

**Implementation Plan**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Documentation Updates Required**:
- [ ] TODO.md - Add task
- [ ] ARCHITECTURE.md - Update section X
- [ ] CHANGELOG.md - Log completion
- [ ] AGENT.md - Update capabilities

**Proceed? (Y/N)**
```

### Bug Report

```markdown
🤖 [FEATURE] AGENT
📋 Template: FEATURE_TEMPLATE.md

**Bug Reported**: [Description]

**Impact**: High/Medium/Low

**Affected Files**:
- [file_path.py:line_number]

**Fix Plan**:
1. [Diagnosis]
2. [Solution]
3. [Testing]

**Documentation Updates**:
- [ ] TODO.md → Move from Known Issues to Completed
- [ ] CHANGELOG.md → Add to [X.Y.Z] Fixed section
- [ ] README.md → Update troubleshooting (if needed)

**Estimated Fix Time**: [Duration]
```

## Documentation Update Workflow

### When Starting a Task

```markdown
📝 STARTING TASK

Feature: [Feature Name]
Task: [Task Description]

1. Adding to TODO.md:

   ## 🔴 High Priority (Current Sprint)

   ### [Task Name]
   **Description**: [What needs to be done]
   **Reason**: [Why it's important]
   **Effort**: [Time estimate]
   **Assigned**: [Who's working]
   **Status**: 🚧 In Progress

2. Creating branch (if applicable):
   git checkout -b feature/[task-name]

3. Ready to proceed!
```

### When Completing a Task

```markdown
✅ TASK COMPLETED

Feature: [Feature Name]
Task: [Task Description]

1. Updating TODO.md:
   - [x] [Task Name] - COMPLETED

2. Updating CHANGELOG.md:

   ## [Unreleased]

   ### Added
   - **[Feature]**: [What was added]

   Or:

   ### Fixed
   - **[Bug]**: [What was fixed]

3. Updating AGENT.md (if capabilities changed):

   ### What This Agent CAN Do
   - ✅ [New capability added]

4. Version bump needed?
   - Patch (X.Y.Z+1): Bug fix
   - Minor (X.Y+1.0): New feature
   - Major (X+1.0.0): Breaking change

5. Documentation validated: ✅
```

### When Adding a Wishlist Item

```markdown
💡 NEW IDEA

Feature: [Feature Name]
Idea: [Description]

Adding to WISHLIST.md:

## Priority [1-4] ([Timeframe])

### [Enhancement Name]
**Description**: [What it does]
**Value**: [Why it's important]
**Effort**: [Time estimate]
**Dependencies**: [What's needed]
**Requested By**: [User/Team/Internal]

Wishlist updated! ✅
```

## Validation Checklist

Before considering any task complete, verify:

### Documentation Complete
- [ ] All 7 files exist for the feature
- [ ] Files follow FEATURE_TEMPLATE.md structure
- [ ] Cross-references are correct
- [ ] Code examples are tested
- [ ] Screenshots/diagrams are current

### Consistency Maintained
- [ ] Version numbers match across files
- [ ] CHANGELOG.md reflects current version
- [ ] TODO.md reflects actual status
- [ ] AGENT.md capabilities are accurate

### Integration Verified
- [ ] MAIN_AGENT.md knows about feature
- [ ] features/INDEX.md includes feature
- [ ] Dependencies documented in AGENT.md
- [ ] Cross-feature impacts noted

### Quality Standards Met
- [ ] Code compiles without errors
- [ ] Tests pass
- [ ] Performance acceptable
- [ ] Security validated
- [ ] Error handling implemented

## Automated Reminders

### On Every User Message

The Main Agent will display:

```
🤖 MAGNUS MAIN AGENT
📋 Template: FEATURE_TEMPLATE.md

[Response to user]

---
💡 Remember:
- Follow FEATURE_TEMPLATE.md for all documentation
- Update TODO.md for task tracking
- Update CHANGELOG.md for version history
- Maintain uniformity across features
```

### On Feature Creation

```
🎉 CREATING NEW FEATURE: [Name]

📋 Using FEATURE_TEMPLATE.md

Required files to create:
1. ⏳ README.md (User Guide)
2. ⏳ ARCHITECTURE.md (Technical)
3. ⏳ SPEC.md (Requirements)
4. ⏳ WISHLIST.md (Future Plans)
5. ⏳ AGENT.md (Agent System)
6. ⏳ TODO.md (Current Tasks)
7. ⏳ CHANGELOG.md (Version 1.0.0)

Integration tasks:
- ⏳ Register in MAIN_AGENT.md
- ⏳ Add to features/INDEX.md
- ⏳ Update navigation in dashboard.py

Starting feature creation...
```

### On Feature Modification

```
🔧 MODIFYING FEATURE: [Name]

📋 Following FEATURE_TEMPLATE.md

Pre-modification checklist:
✓ Read features/[name]/AGENT.md
✓ Check features/[name]/TODO.md
✓ Review features/[name]/CHANGELOG.md

Making changes...

Post-modification checklist:
- [ ] Update TODO.md (mark complete)
- [ ] Update CHANGELOG.md (log change)
- [ ] Update AGENT.md (if needed)
- [ ] Update version number
- [ ] Test integration

Remember to commit with proper message!
```

## Template Enforcement

### On Missing Files

```
⚠️ INCOMPLETE DOCUMENTATION DETECTED

Feature: [Name]
Location: features/[name]/

Missing files:
❌ [FILE1.md]
❌ [FILE2.md]

Required: All 7 files per FEATURE_TEMPLATE.md

Would you like me to:
1. Create missing files from template
2. Show template for manual creation
3. Continue anyway (not recommended)

Choose an option:
```

### On Format Violations

```
⚠️ DOCUMENTATION FORMAT ISSUE

File: features/[name]/[FILE.md]

Issue: Missing required section "[Section Name]"
Expected: See FEATURE_TEMPLATE.md line X

Fix:
1. Add missing section
2. Follow template structure
3. Validate with checklist

Continue? (Y/N)
```

## Quick Reference

### Main Agent Greeting

Every conversation with Main Agent starts with:

```
🤖 MAGNUS MAIN AGENT
📋 Template: FEATURE_TEMPLATE.md
📊 Status: 10 features, 70 docs, all synchronized

How can I help you today?
```

### Feature Agent Greeting

Every feature-specific conversation starts with:

```
🤖 [FEATURE NAME] AGENT
📋 Template: FEATURE_TEMPLATE.md
📂 features/[name]/ (7/7 docs ✓)

Ready to assist!
```

### Reminder Footer

Every response ends with:

```
---
💡 Documentation Reminder:
• Follow FEATURE_TEMPLATE.md
• Update TODO.md, CHANGELOG.md, WISHLIST.md
• Maintain uniformity across all features
```

## Implementation Notes

This protocol should be:
1. **Displayed prominently** in every agent interaction
2. **Enforced automatically** through validation
3. **Updated regularly** as processes improve
4. **Referenced frequently** in documentation

## Version History

- **1.0.0** (2025-11-01): Initial protocol definition

---

**This protocol ensures Magnus maintains world-class documentation quality and consistency.**

For template details, see: [FEATURE_TEMPLATE.md](FEATURE_TEMPLATE.md)
For agent system, see: [MAIN_AGENT.md](MAIN_AGENT.md)
For feature index, see: [features/INDEX.md](features/INDEX.md)
