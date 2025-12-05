#!/usr/bin/env python3
"""
QA Daemon Watchdog

A robust watchdog script that:
1. Ensures the QA daemon is always running
2. Restarts it if it crashes or hangs
3. Monitors health via status.json
4. Sends Telegram alerts on failures
"""

import os
import sys
import time
import json
import subprocess
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Setup logging to file
LOG_FILE = Path(__file__).parent / "logs" / "watchdog.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
QA_RUNNER_PATH = Path(__file__).parent / "qa_runner.py"
STATUS_FILE = Path(__file__).parent / "data" / "status.json"
PID_FILE = Path(__file__).parent / "data" / "watchdog.pid"

# Configuration
DAEMON_INTERVAL_MINUTES = 10
HEALTH_CHECK_INTERVAL_SECONDS = 60  # Check daemon health every minute
STALE_THRESHOLD_MINUTES = 15  # Consider daemon stuck if no update in 15 min
MAX_RESTART_ATTEMPTS = 3
RESTART_BACKOFF_SECONDS = 30


class QAWatchdog:
    """Watchdog that keeps the QA daemon running."""

    def __init__(self) -> None:
        self.daemon_process: Optional[subprocess.Popen] = None
        self.running = False
        self.restart_count = 0
        self.last_restart = None

    def write_pid(self) -> None:
        """Write our PID to file."""
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
        logger.info(f"Watchdog PID: {os.getpid()}")

    def cleanup_pid(self) -> None:
        """Remove PID file on exit."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def is_daemon_running(self) -> bool:
        """Check if daemon process is running."""
        if self.daemon_process is None:
            return False
        return self.daemon_process.poll() is None

    def is_daemon_healthy(self) -> bool:
        """Check if daemon is healthy by reading status.json."""
        if not STATUS_FILE.exists():
            return False

        try:
            status = json.loads(STATUS_FILE.read_text())
            last_updated = status.get("last_updated")

            if not last_updated:
                return False

            # Parse timestamp and check if recent
            try:
                last_update_time = datetime.fromisoformat(last_updated)
                age_minutes = (datetime.now() - last_update_time).total_seconds() / 60

                if age_minutes > STALE_THRESHOLD_MINUTES:
                    logger.warning(f"Status file is stale ({age_minutes:.1f} min old)")
                    return False

                return True
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse status timestamp: {e}")
                return False

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not read status file: {e}")
            return False

    def start_daemon(self) -> bool:
        """Start the QA daemon process."""
        try:
            logger.info("Starting QA daemon...")

            # Use pythonw.exe on Windows for windowless execution
            python_exe = sys.executable
            if sys.platform == "win32":
                pythonw = Path(python_exe).parent / "pythonw.exe"
                if pythonw.exists():
                    python_exe = str(pythonw)

            # Start daemon with output to log file
            log_file = Path(__file__).parent / "logs" / "daemon.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as f:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"Daemon started at {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n\n")

            self.daemon_process = subprocess.Popen(
                [
                    python_exe,
                    str(QA_RUNNER_PATH),
                    "--daemon",
                    "--force",  # Always force start to overcome stale locks
                    "--interval", str(DAEMON_INTERVAL_MINUTES),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            self.last_restart = datetime.now()
            logger.info(f"Daemon started with PID: {self.daemon_process.pid}")

            # Wait a moment and check if it's still running
            time.sleep(5)
            if not self.is_daemon_running():
                logger.error("Daemon died immediately after start")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False

    def stop_daemon(self) -> None:
        """Stop the daemon process."""
        if self.daemon_process:
            logger.info("Stopping daemon...")
            try:
                self.daemon_process.terminate()
                self.daemon_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Daemon did not terminate gracefully, killing...")
                self.daemon_process.kill()
            except Exception as e:
                logger.error(f"Error stopping daemon: {e}")
            finally:
                self.daemon_process = None

    def restart_daemon(self) -> None:
        """Restart the daemon with backoff."""
        self.restart_count += 1

        if self.restart_count > MAX_RESTART_ATTEMPTS:
            # Check if enough time has passed since last restart
            if self.last_restart:
                time_since_restart = (datetime.now() - self.last_restart).total_seconds()
                if time_since_restart < RESTART_BACKOFF_SECONDS * MAX_RESTART_ATTEMPTS:
                    logger.error(
                        f"Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached. "
                        f"Waiting before retry..."
                    )
                    time.sleep(RESTART_BACKOFF_SECONDS * MAX_RESTART_ATTEMPTS)
                    self.restart_count = 0

        # Stop existing daemon
        self.stop_daemon()

        # Wait before restart
        backoff = RESTART_BACKOFF_SECONDS * min(self.restart_count, 3)
        logger.info(f"Waiting {backoff}s before restart (attempt {self.restart_count})...")
        time.sleep(backoff)

        # Start daemon
        if self.start_daemon():
            logger.info("Daemon restarted successfully")
            # Reset counter on successful start + health check
            time.sleep(30)  # Wait for first health update
            if self.is_daemon_healthy():
                self.restart_count = 0
        else:
            logger.error("Failed to restart daemon")

    def run(self) -> None:
        """Main watchdog loop."""
        self.running = True
        self.write_pid()

        logger.info("="*60)
        logger.info("QA Watchdog starting")
        logger.info(f"Daemon interval: {DAEMON_INTERVAL_MINUTES} minutes")
        logger.info(f"Health check interval: {HEALTH_CHECK_INTERVAL_SECONDS} seconds")
        logger.info("="*60)

        # Start daemon initially
        if not self.start_daemon():
            logger.error("Initial daemon start failed!")

        try:
            while self.running:
                time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

                # Check if daemon process is running
                if not self.is_daemon_running():
                    logger.warning("Daemon process not running!")
                    self.restart_daemon()
                    continue

                # Check if daemon is healthy (updating status)
                if not self.is_daemon_healthy():
                    logger.warning("Daemon appears unhealthy (stale status)")
                    self.restart_daemon()
                    continue

                # Daemon is healthy
                if self.restart_count > 0:
                    logger.info("Daemon recovered, resetting restart count")
                    self.restart_count = 0

        except KeyboardInterrupt:
            logger.info("Watchdog shutdown requested")
        finally:
            self.stop_daemon()
            self.cleanup_pid()
            self.running = False
            logger.info("Watchdog stopped")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="QA Daemon Watchdog")
    parser.add_argument("--stop", action="store_true", help="Stop existing watchdog")
    args = parser.parse_args()

    if args.stop:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text())
            logger.info(f"Stopping watchdog (PID: {pid})")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            PID_FILE.unlink()
        else:
            logger.info("No watchdog PID file found")
        return

    # Check if already running
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text())
        try:
            os.kill(old_pid, 0)  # Check if process exists
            logger.warning(f"Watchdog already running (PID: {old_pid})")
            logger.info("Use --stop to stop it first")
            return
        except OSError:
            # Process doesn't exist, clean up stale PID
            PID_FILE.unlink()

    watchdog = QAWatchdog()
    watchdog.run()


if __name__ == "__main__":
    main()
