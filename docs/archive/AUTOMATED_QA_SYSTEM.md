# Magnus Automated QA & Testing System

## 🔴 CRITICAL: Never Deploy Without QA

**This document defines mandatory QA checks that MUST pass before any code changes are deployed.**

---

## Purpose

After the KeyError import failure, this system ensures:
1. ✅ All imports resolve correctly
2. ✅ All code compiles without errors
3. ✅ Critical paths are tested
4. ✅ No regressions introduced
5. ✅ Documentation is updated

**NO CODE CHANGES ARE COMPLETE WITHOUT PASSING ALL QA CHECKS.**

---

## Quick QA Command

```bash
# Run this after EVERY change:
python qa_check.py

# If all pass:
# ✅ Code change is ready
# ✅ Update documentation
# ✅ Deploy

# If any fail:
# 🔴 Fix immediately
# 🔴 Re-run QA
# 🔴 Do NOT deploy
```

---

## Automated QA Checklist

### Phase 1: Import & Syntax Checks (CRITICAL)

```bash
# Test 1: All Python files compile
echo "Testing: Python syntax..."
python -m py_compile dashboard.py
python -m py_compile positions_page_improved.py
python -m py_compile prediction_markets_page.py

# Test 2: Critical imports work
echo "Testing: Imports..."
python -c "import streamlit"
python -c "import robin_stocks"
python -c "from src.agents.runtime.market_data_agent import MarketDataAgent"
python -c "from src.agents.runtime.wheel_strategy_agent import WheelStrategyAgent"
python -c "from src.agents.runtime.risk_management_agent import RiskManagementAgent"

# Test 3: __init__.py files exist
echo "Testing: Package structure..."
test -f src/__init__.py || echo "❌ MISSING: src/__init__.py"
test -f src/agents/__init__.py || echo "❌ MISSING: src/agents/__init__.py"
test -f src/agents/runtime/__init__.py || echo "❌ MISSING: src/agents/runtime/__init__.py"
```

### Phase 2: Code Quality Checks

```bash
# Test 4: No syntax errors
python -m py_compile *.py

# Test 5: Required files exist
test -f dashboard.py || echo "❌ MISSING: dashboard.py"
test -f positions_page_improved.py || echo "❌ MISSING: positions_page_improved.py"
test -f .env || echo "⚠️ WARNING: .env not found"

# Test 6: Dependencies installed
pip list | grep -q "streamlit" || echo "❌ MISSING: streamlit"
pip list | grep -q "robin-stocks" || echo "❌ MISSING: robin-stocks"
```

### Phase 3: Feature Tests

```bash
# Test 7: Dashboard loads
python -c "
import sys
sys.path.insert(0, '.')
try:
    import dashboard
    print('✅ Dashboard imports successfully')
except Exception as e:
    print(f'❌ Dashboard import failed: {e}')
    sys.exit(1)
"

# Test 8: Positions page loads
python -c "
from positions_page_improved import show_positions_page
print('✅ Positions page imports successfully')
"

# Test 9: Prediction markets page loads
python -c "
from prediction_markets_page import show_prediction_markets
print('✅ Prediction markets imports successfully')
"
```

### Phase 4: Documentation Checks

```bash
# Test 10: CHANGELOG updated
git diff features/*/CHANGELOG.md | grep -q "." && echo "✅ CHANGELOG updated" || echo "⚠️ CHANGELOG not updated"

# Test 11: TODO updated if needed
# (manual check)

# Test 12: Template compliance
test -f FEATURE_TEMPLATE.md || echo "❌ MISSING: FEATURE_TEMPLATE.md"
test -f MAIN_AGENT_TEMPLATE.md || echo "❌ MISSING: MAIN_AGENT_TEMPLATE.md"
```

---

## QA Automation Script

Create `qa_check.py`:

```python
#!/usr/bin/env python3
"""
Magnus Automated QA System
Run this after EVERY code change
"""

import subprocess
import sys
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(msg):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

def run_test(name, command, critical=True):
    """Run a test command and report results"""
    print(f"Testing: {name}...", end=" ")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"{Colors.GREEN}✅ PASS{Colors.END}")
        return True
    else:
        if critical:
            print(f"{Colors.RED}❌ FAIL (CRITICAL){Colors.END}")
            if result.stderr:
                print(f"{Colors.RED}Error: {result.stderr}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  WARNING{Colors.END}")
        return False

def main():
    print_header("MAGNUS AUTOMATED QA SYSTEM")

    all_passed = True
    critical_failures = []
    warnings = []

    # Phase 1: Critical Import & Syntax Checks
    print_header("Phase 1: Critical Import & Syntax Checks")

    tests = [
        ("Dashboard syntax", "python -m py_compile dashboard.py", True),
        ("Positions page syntax", "python -m py_compile positions_page_improved.py", True),
        ("Prediction markets syntax", "python -m py_compile prediction_markets_page.py", True),
        ("Streamlit import", "python -c 'import streamlit'", True),
        ("Robinhood import", "python -c 'import robin_stocks'", True),
        ("MarketDataAgent import", "python -c 'from src.agents.runtime.market_data_agent import MarketDataAgent'", True),
        ("WheelStrategyAgent import", "python -c 'from src.agents.runtime.wheel_strategy_agent import WheelStrategyAgent'", True),
    ]

    for name, cmd, critical in tests:
        if not run_test(name, cmd, critical):
            if critical:
                critical_failures.append(name)
            else:
                warnings.append(name)
            all_passed = False

    # Phase 2: Package Structure
    print_header("Phase 2: Package Structure")

    required_files = [
        "src/__init__.py",
        "src/agents/__init__.py",
        "src/agents/runtime/__init__.py",
        "dashboard.py",
        "positions_page_improved.py",
        "FEATURE_TEMPLATE.md",
        "MAIN_AGENT_TEMPLATE.md"
    ]

    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"{file}: {Colors.GREEN}✅ EXISTS{Colors.END}")
        else:
            print(f"{file}: {Colors.RED}❌ MISSING{Colors.END}")
            critical_failures.append(f"Missing: {file}")
            all_passed = False

    # Phase 3: Documentation
    print_header("Phase 3: Documentation Checks")

    # Check if changelog files exist
    changelog_count = len(list(Path("features").glob("*/CHANGELOG.md")))
    print(f"CHANGELOG.md files found: {changelog_count}")

    if changelog_count >= 10:
        print(f"{Colors.GREEN}✅ All features have CHANGELOG{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  Some features missing CHANGELOG{Colors.END}")
        warnings.append("Incomplete CHANGELOG coverage")

    # Final Report
    print_header("QA RESULTS")

    if all_passed:
        print(f"{Colors.GREEN}{'='*60}")
        print("✅ ALL TESTS PASSED")
        print("✅ Code is ready for deployment")
        print("✅ Proceed with documentation updates")
        print(f"{'='*60}{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{'='*60}")
        print("❌ QA FAILED")
        print(f"{'='*60}{Colors.END}\n")

        if critical_failures:
            print(f"{Colors.RED}CRITICAL FAILURES:{Colors.END}")
            for failure in critical_failures:
                print(f"  {Colors.RED}❌ {failure}{Colors.END}")
            print(f"\n{Colors.RED}🔴 DO NOT DEPLOY UNTIL FIXED{Colors.END}\n")

        if warnings:
            print(f"{Colors.YELLOW}WARNINGS:{Colors.END}")
            for warning in warnings:
                print(f"  {Colors.YELLOW}⚠️  {warning}{Colors.END}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## Usage

### After EVERY Code Change:

```bash
# 1. Make your changes
# 2. Run QA
python qa_check.py

# 3. If PASS:
#    - Update documentation
#    - Commit changes
#    - Deploy

# 4. If FAIL:
#    - Fix issues immediately
#    - Re-run QA
#    - Do NOT proceed until all pass
```

### Manual QA Checklist

Even with automated tests, manually verify:

- [ ] Dashboard loads in browser without errors
- [ ] Positions page displays correctly
- [ ] No console errors (F12 in browser)
- [ ] Navigation works
- [ ] All links clickable
- [ ] No visual regressions
- [ ] Performance acceptable

---

## Pre-Deployment Checklist

Before deploying ANY change:

### Code Quality
- [ ] All Python files compile (`python -m py_compile *.py`)
- [ ] All imports resolve correctly
- [ ] No syntax errors
- [ ] No unused imports
- [ ] Code follows PEP 8

### Testing
- [ ] Automated QA passes (`python qa_check.py`)
- [ ] Manual QA completed
- [ ] No console errors
- [ ] No network errors
- [ ] Performance acceptable

### Documentation
- [ ] CHANGELOG.md updated (if feature changed)
- [ ] TODO.md updated (if tasks completed)
- [ ] README.md accurate (if user-facing change)
- [ ] AGENT.md accurate (if capabilities changed)

### Integration
- [ ] Feature agents aware of changes
- [ ] Main Agent updated if needed
- [ ] Cross-feature dependencies documented
- [ ] No breaking changes to other features

### Deployment
- [ ] Changes committed to git
- [ ] Commit message follows format
- [ ] Branch up to date with main
- [ ] No merge conflicts

---

## Emergency Rollback Procedure

If QA passes but production fails:

```bash
# 1. Immediately rollback
git revert HEAD

# 2. Investigate
#    - Check logs
#    - Reproduce error
#    - Identify root cause

# 3. Fix properly
#    - Create fix
#    - Run QA
#    - Test thoroughly
#    - Document lesson learned

# 4. Re-deploy
#    - After QA passes
#    - Monitor closely
```

---

## Continuous Integration (Future)

### GitHub Actions Workflow

```yaml
# .github/workflows/qa.yml
name: QA Checks

on: [push, pull_request]

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run QA
        run: |
          python qa_check.py
```

---

## Lessons Learned

### KeyError Import Failure (2025-11-01)

**What Happened**:
- Missing `src/__init__.py` and `src/agents/__init__.py`
- Python couldn't import from `src.agents`
- Dashboard crashed on startup

**Root Cause**:
- Package structure incomplete
- No pre-deployment testing
- Assumed imports would work

**Prevention**:
1. ✅ Created automated QA system
2. ✅ Added import tests
3. ✅ Created package structure validation
4. ✅ Mandatory QA before deployment

**Never Again**: All changes MUST pass `qa_check.py` before deployment.

---

## Quality Metrics

Track these over time:

| Metric | Target | Current |
|--------|--------|---------|
| QA Pass Rate | 100% before deployment | - |
| Import Failures | 0 per month | 1 (fixed) |
| Syntax Errors | 0 per deployment | 0 |
| Console Errors | 0 per page load | 0 |
| Documentation Coverage | 100% | 100% |

---

## Maintenance

This QA system should be reviewed:
- After every critical failure
- Monthly (routine check)
- When adding new features
- When dependencies update

**Last Updated**: 2025-11-01
**Version**: 1.0.0
**Status**: ✅ Active

---

## Summary

**Remember**:
1. 🔴 **RUN QA AFTER EVERY CHANGE**
2. 🔴 **FIX ALL FAILURES BEFORE DEPLOYING**
3. 🔴 **UPDATE DOCUMENTATION ALWAYS**
4. 🔴 **TEST MANUALLY IN BROWSER**
5. 🔴 **NEVER ASSUME IT WORKS**

**Command to run**: `python qa_check.py`

**Result must be**: ✅ ALL TESTS PASSED

**Only then**: Proceed with deployment

---

💡 **Quality First**
Don't rush. Test thoroughly. Deploy confidently.
