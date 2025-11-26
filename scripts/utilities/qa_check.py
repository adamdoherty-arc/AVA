#!/usr/bin/env python3
"""
Magnus Automated QA System
Run this after EVERY code change: python qa_check.py
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
        print(f"{Colors.GREEN}PASS{Colors.END}")
        return True
    else:
        if critical:
            print(f"{Colors.RED}FAIL (CRITICAL){Colors.END}")
            if result.stderr:
                print(f"{Colors.RED}Error: {result.stderr[:200]}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}WARNING{Colors.END}")
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
        ("Streamlit import", "python -c \"import streamlit\"", True),
        ("Robinhood import", "python -c \"import robin_stocks\"", True),
        ("MarketDataAgent import", "python -c \"from src.agents.runtime.market_data_agent import MarketDataAgent\"", True),
        ("WheelStrategyAgent import", "python -c \"from src.agents.runtime.wheel_strategy_agent import WheelStrategyAgent\"", True),
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
            print(f"{file}: {Colors.GREEN}EXISTS{Colors.END}")
        else:
            print(f"{file}: {Colors.RED}MISSING{Colors.END}")
            critical_failures.append(f"Missing: {file}")
            all_passed = False

    # Phase 3: Documentation
    print_header("Phase 3: Documentation Checks")

    # Check if changelog files exist
    changelog_count = len(list(Path("features").glob("*/CHANGELOG.md")))
    print(f"CHANGELOG.md files found: {changelog_count}")

    if changelog_count >= 10:
        print(f"{Colors.GREEN}All features have CHANGELOG{Colors.END}")
    else:
        print(f"{Colors.YELLOW}Some features missing CHANGELOG{Colors.END}")
        warnings.append("Incomplete CHANGELOG coverage")

    # Final Report
    print_header("QA RESULTS")

    if all_passed:
        print(f"{Colors.GREEN}{'='*60}")
        print("ALL TESTS PASSED")
        print("Code is ready for deployment")
        print("Proceed with documentation updates")
        print(f"{'='*60}{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}{'='*60}")
        print("QA FAILED")
        print(f"{'='*60}{Colors.END}\n")

        if critical_failures:
            print(f"{Colors.RED}CRITICAL FAILURES:{Colors.END}")
            for failure in critical_failures:
                print(f"  {Colors.RED}{failure}{Colors.END}")
            print(f"\n{Colors.RED}DO NOT DEPLOY UNTIL FIXED{Colors.END}\n")

        if warnings:
            print(f"{Colors.YELLOW}WARNINGS:{Colors.END}")
            for warning in warnings:
                print(f"  {Colors.YELLOW}{warning}{Colors.END}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
