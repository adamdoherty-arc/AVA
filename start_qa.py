#!/usr/bin/env python3
"""
Magnus QA System Startup Script

Usage:
    python start_qa.py --once      # Run once
    python start_qa.py --daemon    # Run continuously every 20 minutes
    python start_qa.py --visible   # Show console output
"""

import sys
import os
from pathlib import Path

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".claude" / "orchestrator"))

# Set working directory
os.chdir(PROJECT_ROOT)

# Now import and run
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Magnus QA System')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--daemon', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=20, help='Interval in minutes')
    parser.add_argument('--visible', action='store_true', help='Show console output')

    args = parser.parse_args()

    # Import the QA runner
    from continuous_qa.qa_runner import QARunner

    runner = QARunner(
        interval_minutes=args.interval,
        visible=args.visible or not args.daemon or args.once,
    )

    if args.once:
        print("=" * 60)
        print("MAGNUS QA SYSTEM - Single Run")
        print("=" * 60)
        result = runner.run_once()
        print("\n" + "=" * 60)
        print(f"Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
        print(f"Health Score: {result.get('health_score', 'N/A')}")
        print(f"Issues Found: {result.get('issues_found', 0)}")
        print(f"Issues Fixed: {result.get('issues_fixed', 0)}")
        print("=" * 60)
        sys.exit(0 if result.get('success', False) else 1)
    elif args.daemon:
        print("=" * 60)
        print(f"MAGNUS QA SYSTEM - Continuous Mode (every {args.interval} min)")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        runner.run_continuous()
    else:
        # Default: run once with visible output
        print("=" * 60)
        print("MAGNUS QA SYSTEM - Single Run")
        print("=" * 60)
        result = runner.run_once()
        print("\n" + "=" * 60)
        print(f"Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
        print(f"Health Score: {result.get('health_score', 'N/A')}")
        print("=" * 60)
        sys.exit(0 if result.get('success', False) else 1)
