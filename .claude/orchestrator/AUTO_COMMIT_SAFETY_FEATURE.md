# Auto-Commit Safety Feature

## Overview
The orchestrator now automatically commits QA-validated changes to prevent data loss from git operations gone wrong.

Additionally, the orchestrator supports a **Plan-First Workflow** that researches current implementation and modern patterns before execution, presenting a detailed plan for user approval.

## Plan-First Workflow (NEW)

### What Is It?
The Plan-First Workflow ensures the orchestrator never executes blindly. Before making any code changes:

1. **Researches Current Code** - Analyzes existing implementation using Grep/Glob/Read
2. **Researches Modern Patterns** - Searches GitHub, frameworks, and best practices
3. **Generates Detailed Plan** - Creates step-by-step execution plan with file paths and line numbers
4. **Awaits User Approval** - Presents plan via ExitPlanMode before execution

### Configuration
In `.claude/orchestrator/config.yaml`:
```yaml
orchestrator:
  enabled: true
  mode: standard
  auto_run_on_request: false  # REQUIRED: Disable auto-execution
  require_plan_approval: true  # Enable plan-first workflow
  plan_research_enabled: true  # Enable research phase
  research_sources:
    - current_codebase       # Analyze existing code
    - github_projects        # Search similar implementations
    - modern_frameworks      # Look up 2025 patterns
    - documentation          # Check official docs
  research_depth: medium     # quick, medium, deep
```

### Research Phases

#### Phase 1: Current Codebase Analysis
```python
# Extracts keywords from request
# Uses Grep/Glob to find relevant files
# Reads and analyzes existing implementation
# Identifies patterns and structure
```

#### Phase 2: GitHub Research
```python
# Generates search queries based on request
# Searches GitHub for similar implementations
# Identifies common patterns and best practices
# Extracts relevant code examples
```

#### Phase 3: Framework Research
```python
# Searches for 2025 best practices
# Looks up modern framework patterns
# Finds architecture recommendations
# Identifies anti-patterns to avoid
```

#### Phase 4: Best Practices Research
```python
# Searches industry best practices
# Identifies common pitfalls
# Finds performance optimization tips
# Discovers security considerations
```

#### Phase 5: Plan Synthesis
```python
# Combines all research findings
# Generates detailed execution plan
# Identifies risks and considerations
# Creates test strategy
```

### Plan Output Format
The generated plan includes:
```markdown
# DETAILED EXECUTION PLAN

## Request
[User's original request]

## Current Implementation
Files involved: file1.py, file2.py
[Analysis of existing code]

## Modern Patterns to Apply
- Pattern 1: [Description]
- Pattern 2: [Description]

## Best Practices
- Practice 1: [Description]
- Practice 2: [Description]

## Implementation Steps
1. [Detailed step with file:line numbers]
2. [Detailed step with file:line numbers]

## Risks
- RISK: [Potential issue]

## Test Strategy
- Unit tests: [What to test]
- Integration tests: [What to test]
- Manual testing: [What to verify]
```

### Research Depth Levels

**Quick (depth: quick)**
- 1-2 search queries per source
- Minimal analysis
- Fast turnaround
- Good for simple changes

**Medium (depth: medium)** - RECOMMENDED
- 3-5 search queries per source
- Balanced analysis
- Good coverage
- Default setting

**Deep (depth: deep)**
- 5-10 search queries per source
- Comprehensive analysis
- Thorough research
- Use for complex features

### Benefits

1. **No Blind Execution** - Orchestrator always understands the context
2. **Modern Patterns** - Applies 2025 best practices automatically
3. **Risk Awareness** - Identifies potential issues before coding
4. **User Control** - You approve before any changes are made
5. **Learning** - Orchestrator learns from real-world implementations

### Example Workflow
```
User: "Fix the subscription feature"
↓
Orchestrator: Research Phase
  - Analyzes game_watchlist_manager.py
  - Searches "subscription watchlist dashboard python streamlit"
  - Searches "subscription management best practices 2025"
  - Finds modern patterns and examples
↓
Orchestrator: Plan Generation
  - Current state: Fragmented UI across 3 files
  - Proposed: Consolidated subscription_management_page.py
  - Modern patterns: Card-based layout, search/filter, status badges
  - Risks: Breaking existing integrations
  - Test strategy: End-to-end subscription flow
↓
Orchestrator: Present Plan (ExitPlanMode)
  - Shows detailed plan to user
  - Waits for approval
↓
User: Approves plan
↓
Orchestrator: Execution
  - Follows plan step by step
  - Makes changes with full context
  - Runs QA validation
  - Auto-commits if QA passes
↓
User: Tests changes
  - Runs application
  - Verifies functionality
  - Checks edge cases
↓
User: Push to GitHub
  - git push origin main
  - Changes now available to team
```

### Disabling Plan-First Workflow
If you want orchestrator to execute immediately (old behavior):
```yaml
orchestrator:
  auto_run_on_request: true  # Execute immediately
  require_plan_approval: false  # Skip plan phase
```

## How It Works

### 1. Automatic Execution
After every code change, the orchestrator:
1. ✅ Runs QA validation checks
2. ✅ If QA passes → Auto-commits the changes
3. ✅ If QA fails → Leaves changes uncommitted for manual review
4. ✅ After testing → Push to GitHub to share with team

**Push to GitHub After QA:**
Once code changes are complete, QA validated, and tested:
```bash
# Review what will be pushed
git log origin/main..HEAD

# Push to remote repository
git push origin main

# Or if you're on a feature branch
git push origin your-branch-name
```

**When to Push:**
- ✅ QA validation passed
- ✅ Manual testing completed
- ✅ All tests passing
- ✅ Code reviewed (if required)
- ✅ Ready for team to use

### 2. Commit Message Format
```
chore: <description> (<features>)

✅ QA Validation: PASSED
📁 Files modified: <count>

Modified files:
  - file1.py
  - file2.py

QA checks run: <checks>

🤖 Auto-committed by Orchestrator to prevent data loss

Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 3. Safety Features
- **No destructive commits**: Uses `--no-verify` to skip hooks if needed
- **Git repo detection**: Only commits if in a valid git repository
- **Change verification**: Verifies changes exist before committing
- **Timeout protection**: All git operations have timeouts to prevent hangs
- **Error logging**: All errors are logged for debugging

## Configuration

### Enable/Disable
In `.claude/orchestrator/config.yaml`:
```yaml
post_execution:
  enabled: true
  auto_commit: true  # Set to false to disable auto-commit
```

### When Auto-Commit Happens
Auto-commit triggers when:
- ✅ Post-execution QA is enabled
- ✅ QA validation passes
- ✅ `auto_commit` config is `true` (default)
- ✅ Changes are detected in the working directory

### When Auto-Commit Does NOT Happen
- ❌ QA validation fails
- ❌ `auto_commit` is disabled
- ❌ Not in a git repository
- ❌ No changes detected
- ❌ Git operations fail

## Benefits

### 1. Data Loss Prevention
**Problem Solved**: Git operations like `reset --hard` can destroy uncommitted work.

**Solution**: Every QA-validated change is immediately committed, creating a safety checkpoint.

### 2. Automatic Recovery Points
Each auto-commit creates a recovery point. If something goes wrong:
```bash
# View recent auto-commits
git log --oneline | grep "Auto-committed by Orchestrator"

# Recover from a specific commit
git show <commit-sha>:file.py > file.py
```

### 3. Complete Audit Trail
Every change is tracked with:
- Timestamp
- QA validation status
- Files modified
- Feature context

## Examples

### Example 1: Normal Workflow
```
User: "Fix the theta decay dropdown overlap"
↓
Claude makes changes to 2 files
↓
Orchestrator runs QA → PASSED
↓
Auto-commit: "chore: Fix theta decay dropdown overlap"
↓
Changes safely in git history
↓
User tests changes → Everything works
↓
User: git push origin main
↓
Changes now on GitHub for team to use
```

### Example 2: QA Failure
```
User: "Add horizontal lines for spacing"
↓
Claude makes changes
↓
Orchestrator runs QA → FAILED (horizontal lines forbidden)
↓
NO auto-commit (changes stay uncommitted)
↓
User can manually review and fix
```

### Example 3: Recovery After Git Reset
```
Git operation destroys work
↓
Check reflog and dangling commits:
  git reflog
  git fsck --lost-found
↓
Find auto-commit with changes:
  git show <commit-sha>
↓
Recover files:
  git cherry-pick <commit-sha>
```

## Manual Operations

### View Auto-Commit History
```bash
# Show all auto-commits
git log --oneline --grep="Auto-committed by Orchestrator"

# Show details of latest auto-commit
git log --grep="Auto-committed by Orchestrator" -1 --stat
```

### Disable Auto-Commit Temporarily
```python
from .claude.orchestrator.main_orchestrator import get_orchestrator

orchestrator = get_orchestrator()
orchestrator.config["post_execution"]["auto_commit"] = False
```

### Test Auto-Commit
```bash
cd .claude/orchestrator
python main_orchestrator.py --qa file1.py file2.py
```

## Best Practices

### 1. Keep Auto-Commit Enabled
Unless you have a specific reason, keep `auto_commit: true`. It's a safety net.

### 2. Review Auto-Commits Periodically
Check auto-commit messages to understand what changes were validated:
```bash
git log --oneline --grep="Auto-committed" | head -10
```

### 3. Squash Before Push
Before pushing to remote, squash auto-commits into meaningful commits:
```bash
git rebase -i HEAD~10  # Squash last 10 commits
```

### 4. Always Push After Testing
Once QA passes and you've tested the changes, **push to GitHub immediately**:
```bash
# Review what you're pushing
git log origin/main..HEAD
git diff origin/main..HEAD

# Test one final time
python dashboard.py  # or your test command

# Push to GitHub
git push origin main
```

**Why push immediately?**
- ✅ Backs up your work to remote
- ✅ Makes changes available to your team
- ✅ Triggers CI/CD pipelines
- ✅ Creates a shared checkpoint
- ✅ Prevents lost work if local machine fails

**Before pushing, verify:**
- ✅ QA validation passed
- ✅ Manual testing completed successfully
- ✅ No sensitive data in commits (API keys, passwords, etc.)
- ✅ Commit messages are clear and descriptive

### 5. Trust QA Validation
Auto-commits only happen when QA passes. If QA is passing bad code:
- Fix the QA checks
- Don't disable auto-commit

## Troubleshooting

### Auto-Commit Not Working?
Check:
1. Is `post_execution.auto_commit` set to `true` in config?
2. Is QA validation passing?
3. Are you in a git repository?
4. Check logs: `.claude/orchestrator/orchestrator.log`

### Too Many Auto-Commits?
This is normal and safe. Squash before pushing:
```bash
git rebase -i origin/main
```

### Want to Disable?
Set in config:
```yaml
post_execution:
  auto_commit: false
```

## Technical Details

### Implementation
See: `.claude/orchestrator/main_orchestrator.py`
- `post_execution_qa()` - Triggers auto-commit
- `_auto_commit_changes()` - Handles git operations
- `_generate_commit_message()` - Creates descriptive messages

### Git Commands Used
```bash
git rev-parse --git-dir        # Check if in repo
git status --porcelain         # Check for changes
git add <files>                # Stage files
git commit --no-verify -m ""   # Create commit
git rev-parse HEAD             # Get commit SHA
```

### Error Handling
All git operations have:
- 5-10 second timeouts
- Capture output for logging
- Exception handling
- Graceful failure (doesn't crash orchestrator)

## FAQ

**Q: Will this create too many commits?**
A: Yes, but that's intentional. Each commit is a safety checkpoint. Squash before pushing.

**Q: Can I customize commit messages?**
A: Yes, modify `_generate_commit_message()` in `main_orchestrator.py`.

**Q: What if auto-commit fails?**
A: Changes remain uncommitted. Check logs for the error. Your work is still safe.

**Q: Does this affect git hooks?**
A: Uses `--no-verify` to skip hooks. Pre-commit hooks shouldn't block safety commits.

**Q: Can I recover from `git reset --hard`?**
A: Yes! Auto-commits are still in reflog and fsck can find dangling commits.

**Q: When should I push to GitHub?**
A: **Immediately after testing!** Once QA passes and you verify changes work:
1. Review commits: `git log origin/main..HEAD`
2. Test one final time
3. Push: `git push origin main`

Don't wait - push creates a backup and shares with team. You can always revert if needed.

**Q: Should I push auto-commits or squash first?**
A: Depends on your team's workflow:
- **Squash first** (recommended): Creates clean commit history
- **Push as-is**: Faster, but creates many commits. Clean up later with `git rebase -i`

**Q: What if I forget to push?**
A: Your local commits are safe but:
- Team can't see your changes
- Work isn't backed up to remote
- CI/CD doesn't run
Set a reminder or create an alias: `alias pushit='git push origin main'`

## Lessons Learned

This feature was added after a `git reset --hard` destroyed hours of uncommitted work. The orchestrator had to use `git fsck --lost-found` to recover dangling commits from a stash.

**Never again.** Every QA-validated change is now safely committed.

## See Also
- [Orchestrator Configuration](config.yaml)
- [QA Agent](qa_agent.py)
- [Rule Engine](rule_engine.py)
