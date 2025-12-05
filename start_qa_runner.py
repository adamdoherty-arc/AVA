#!/usr/bin/env python3
"""
Magnus QA Runner Startup Script

This script properly sets up the Python path and runs the QA runner.

Usage:
    python start_qa_runner.py --once --visible
    python start_qa_runner.py --daemon --interval 10
"""

import os
import sys
import argparse

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Add necessary paths
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '.claude', 'orchestrator', 'continuous_qa'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))


def main():
    parser = argparse.ArgumentParser(description='Magnus QA Runner')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--daemon', action='store_true',
                        help='Run continuously')
    parser.add_argument('--interval', type=int, default=10,
                        help='Interval between runs in minutes (default: 10)')
    parser.add_argument('--visible', action='store_true',
                        help='Show visible console output')
    parser.add_argument('--force', action='store_true',
                        help='Force start (kill existing instance)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of QA runner')
    parser.add_argument('--stop', action='store_true',
                        help='Stop running QA runner')

    args = parser.parse_args()

    # Import process manager
    try:
        from src.utils.process_manager import ProcessManager, stop_service
    except ImportError:
        print("Warning: ProcessManager not available, running without singleton enforcement")
        ProcessManager = None

    # Handle status command
    if args.status:
        if ProcessManager:
            pm = ProcessManager('qa_runner')
            status = pm.get_status()
            print(f"\nQA Runner Status:")
            print(f"  Running: {status['is_running']}")
            print(f"  PID: {status['pid'] or 'N/A'}")
            print(f"  PID File: {status['pid_file']}")
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Handle stop command
    if args.stop:
        if ProcessManager:
            stop_service('qa_runner')
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Import and run the QA runner
    from qa_runner import QARunner

    # Enforce singleton for daemon mode
    process_manager = None
    if args.daemon and ProcessManager:
        pm = ProcessManager('qa_runner')
        is_running, existing_pid = pm.is_already_running()

        if is_running and not args.force:
            print(f"\nERROR: QA Runner is already running (PID {existing_pid})")
            print(f"\nOptions:")
            print(f"  1. Stop existing:  python start_qa_runner.py --stop")
            print(f"  2. Force restart:  python start_qa_runner.py --daemon --force")
            print(f"  3. Check status:   python start_qa_runner.py --status")
            sys.exit(1)

        if not pm.acquire_lock(force=args.force):
            print("Failed to acquire lock")
            sys.exit(1)

        process_manager = pm

    runner = QARunner(
        interval_minutes=args.interval,
        visible=args.visible or not args.daemon,
    )

    try:
        if args.once:
            print(f"\n{'='*60}")
            print("  Magnus QA Runner - Single Execution")
            print(f"{'='*60}\n")
            result = runner.run_once()
            print(f"\nResult: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
            print(f"  Issues Found: {result.get('issues_found', 0)}")
            print(f"  Issues Fixed: {result.get('issues_fixed', 0)}")
            print(f"  Health Score: {result.get('health_score', 0):.1f}/100")
            sys.exit(0 if result.get('success', False) else 1)
        elif args.daemon:
            print(f"\n{'='*60}")
            print("  Magnus QA Runner - Daemon Mode")
            print(f"  Interval: Every {args.interval} minutes")
            print(f"{'='*60}\n")
            runner.run_continuous()
        else:
            # Default: run once with visible output
            runner.visible = True
            print(f"\n{'='*60}")
            print("  Magnus QA Runner - Single Execution")
            print(f"{'='*60}\n")
            result = runner.run_once()
            print(f"\nResult: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
            sys.exit(0 if result.get('success', False) else 1)
    finally:
        if process_manager:
            process_manager.release_lock()


if __name__ == "__main__":
    main()
