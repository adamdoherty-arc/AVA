"""
Process Manager - Singleton enforcement for services

Provides:
- PID file locking to prevent multiple instances
- Port checking to detect running services
- Graceful shutdown/takeover mechanisms
- Cross-platform support (Windows + Linux)
- Context manager support
- Signal handling for graceful shutdown
- HTTP health checks for web services
- Process tree killing

Usage:
    from src.utils.process_manager import ProcessManager

    # As context manager (recommended)
    with ProcessManager('backend', port=8002) as pm:
        run_server()

    # Manual management
    pm = ProcessManager('qa_runner')
    if not pm.acquire_lock():
        print("Another instance is running")
        sys.exit(1)
    try:
        run_service()
    finally:
        pm.release_lock()
"""

from __future__ import annotations

import json
import os
import sys
import signal
import socket
import logging
import atexit
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# PID files location
PID_DIR = Path(__file__).parent.parent.parent / ".pids"

# Default timeouts
DEFAULT_GRACEFUL_TIMEOUT = 5  # seconds to wait for graceful shutdown
DEFAULT_HEALTH_CHECK_TIMEOUT = 2  # seconds for HTTP health check


@dataclass
class ServiceStatus:
    """Status information for a service."""
    service_name: str
    is_running: bool
    pid: Optional[int]
    port: Optional[int]
    port_in_use: bool
    pid_file: str
    current_pid: Optional[int]
    start_time: Optional[str]
    uptime_seconds: Optional[float]
    health_check_url: Optional[str]
    health_ok: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PidFileData:
    """Data stored in PID file."""
    pid: int
    service_name: str
    port: Optional[int]
    start_time: str
    command: str

    @classmethod
    def from_json(cls, data: str) -> 'PidFileData':
        """Parse from JSON string."""
        d = json.loads(data)
        return cls(
            pid=d['pid'],
            service_name=d['service_name'],
            port=d.get('port'),
            start_time=d['start_time'],
            command=d.get('command', 'unknown')
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)


class ProcessManager:
    """
    Manages singleton enforcement for services.

    Uses PID files and port checking to ensure only one instance runs.
    Supports context manager protocol for automatic cleanup.
    """

    def __init__(
        self,
        service_name: str,
        port: Optional[int] = None,
        health_check_path: str = "/api/health",
        graceful_timeout: int = DEFAULT_GRACEFUL_TIMEOUT
    ):
        """
        Initialize process manager.

        Args:
            service_name: Unique name for this service (e.g., 'qa_runner', 'backend')
            port: Optional port number to check/manage
            health_check_path: HTTP path for health checks (default: /api/health)
            graceful_timeout: Seconds to wait for graceful shutdown before force kill
        """
        self.service_name = service_name
        self.port = port
        self.health_check_path = health_check_path
        self.graceful_timeout = graceful_timeout
        self.pid_file = PID_DIR / f"{service_name}.pid"
        self._lock_acquired = False
        self._original_sigterm = None
        self._original_sigint = None
        self._shutdown_requested = False
        self._shutdown_callbacks: list[Callable] = []

        # Ensure PID directory exists
        PID_DIR.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> 'ProcessManager':
        """Context manager entry - acquire lock."""
        if not self.acquire_lock():
            raise RuntimeError(f"{self.service_name} is already running")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - release lock."""
        self.release_lock()
        return False  # Don't suppress exceptions

    def add_shutdown_callback(self, callback: Callable) -> None:
        """Add a callback to be called during graceful shutdown."""
        self._shutdown_callbacks.append(callback)

    def is_already_running(self) -> Tuple[bool, Optional[int]]:
        """
        Check if another instance of this service is running.

        Returns:
            Tuple of (is_running, pid) where pid is the running process ID if found
        """
        # Check PID file
        if self.pid_file.exists():
            try:
                content = self.pid_file.read_text().strip()
                pid_data = self._parse_pid_file(content)
                if pid_data and self._is_process_running(pid_data.pid):
                    return True, pid_data.pid
                else:
                    # Stale PID file, clean it up
                    logger.info(f"Cleaning up stale PID file for {self.service_name}")
                    self.pid_file.unlink()
            except (ValueError, OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error reading PID file: {e}")
                self.pid_file.unlink(missing_ok=True)

        # Also check port if specified
        if self.port and self.is_port_in_use():
            pid = self._get_port_process()
            if pid:
                return True, pid

        return False, None

    def _parse_pid_file(self, content: str) -> Optional[PidFileData]:
        """Parse PID file content (supports both legacy and new JSON format)."""
        content = content.strip()
        if not content:
            return None

        # Try JSON format first
        if content.startswith('{'):
            return PidFileData.from_json(content)

        # Legacy format: just PID
        try:
            pid = int(content)
            return PidFileData(
                pid=pid,
                service_name=self.service_name,
                port=self.port,
                start_time='unknown',
                command='unknown'
            )
        except ValueError:
            return None

    def is_port_in_use(self) -> bool:
        """Check if the configured port is in use."""
        if not self.port:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', self.port))
                return result == 0
        except Exception:
            return False

    def check_health(self) -> Tuple[bool, Optional[str]]:
        """
        Perform HTTP health check on the service.

        Returns:
            Tuple of (is_healthy, response_or_error)
        """
        if not self.port:
            return False, "No port configured"

        url = f"http://localhost:{self.port}{self.health_check_path}"
        try:
            with urlopen(url, timeout=DEFAULT_HEALTH_CHECK_TIMEOUT) as response:
                return True, response.read().decode('utf-8')
        except URLError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def acquire_lock(self, force: bool = False) -> bool:
        """
        Acquire the singleton lock for this service.

        Args:
            force: If True, kill any existing instance first

        Returns:
            True if lock acquired, False if another instance is running
        """
        is_running, existing_pid = self.is_already_running()

        if is_running:
            if force:
                logger.info(f"Force mode: killing existing {self.service_name} (PID {existing_pid})")
                self.kill_existing(existing_pid, graceful=True)
                # Wait a bit for port to be released
                time.sleep(0.5)
            else:
                logger.error(
                    f"{self.service_name} is already running (PID {existing_pid}). "
                    f"Use --force to kill existing instance."
                )
                return False

        # Write our PID with metadata
        try:
            pid_data = PidFileData(
                pid=os.getpid(),
                service_name=self.service_name,
                port=self.port,
                start_time=datetime.now().isoformat(),
                command=' '.join(sys.argv)
            )
            self.pid_file.write_text(pid_data.to_json())
            self._lock_acquired = True

            # Register cleanup on exit
            atexit.register(self._cleanup)

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            logger.info(f"Acquired lock for {self.service_name} (PID {os.getpid()})")
            return True

        except Exception as e:
            logger.error(f"Failed to write PID file: {e}")
            return False

    def release_lock(self) -> None:
        """Release the singleton lock."""
        if self._lock_acquired:
            self._cleanup()
            self._restore_signal_handlers()

    def kill_existing(self, pid: Optional[int] = None, graceful: bool = True) -> None:
        """
        Kill an existing instance of this service.

        Args:
            pid: PID to kill, or None to find from PID file
            graceful: If True, try graceful shutdown first
        """
        if pid is None:
            _, pid = self.is_already_running()

        if pid:
            if graceful:
                self._graceful_kill(pid)
            else:
                self._kill_process(pid)

        # Also kill by port if specified
        if self.port:
            self.kill_port_process(graceful=graceful)

        # Clean up PID file
        self.pid_file.unlink(missing_ok=True)

    def _graceful_kill(self, pid: int) -> None:
        """Attempt graceful shutdown, falling back to force kill."""
        logger.info(f"Attempting graceful shutdown of PID {pid}")

        # Send SIGTERM (or equivalent on Windows)
        if sys.platform == 'win32':
            # On Windows, try to send CTRL_C_EVENT first
            try:
                import subprocess
                # Generate a Ctrl+C event
                subprocess.run(
                    ['taskkill', '/PID', str(pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
            except Exception:
                pass
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        # Wait for process to exit
        start = time.time()
        while time.time() - start < self.graceful_timeout:
            if not self._is_process_running(pid):
                logger.info(f"Process {pid} terminated gracefully")
                return
            time.sleep(0.1)

        # Force kill if still running
        logger.warning(f"Process {pid} didn't terminate gracefully, force killing")
        self._kill_process(pid)

    def kill_port_process(self, graceful: bool = True) -> None:
        """Kill whatever process is using our port."""
        if not self.port:
            return

        pid = self._get_port_process()
        if pid:
            logger.info(f"Killing process {pid} on port {self.port}")
            if graceful:
                self._graceful_kill(pid)
            else:
                self._kill_process(pid)

    def get_status(self) -> ServiceStatus:
        """Get comprehensive status information about this service."""
        is_running, pid = self.is_already_running()
        port_in_use = self.is_port_in_use() if self.port else False

        # Get start time and uptime from PID file
        start_time = None
        uptime_seconds = None
        if self.pid_file.exists():
            try:
                content = self.pid_file.read_text().strip()
                pid_data = self._parse_pid_file(content)
                if pid_data and pid_data.start_time != 'unknown':
                    start_time = pid_data.start_time
                    try:
                        start_dt = datetime.fromisoformat(start_time)
                        uptime_seconds = (datetime.now() - start_dt).total_seconds()
                    except ValueError:
                        pass
            except Exception:
                pass

        # Check health
        health_url = None
        health_ok = None
        if self.port and is_running:
            health_url = f"http://localhost:{self.port}{self.health_check_path}"
            health_ok, _ = self.check_health()

        return ServiceStatus(
            service_name=self.service_name,
            is_running=is_running,
            pid=pid,
            port=self.port,
            port_in_use=port_in_use,
            pid_file=str(self.pid_file),
            current_pid=os.getpid() if self._lock_acquired else None,
            start_time=start_time,
            uptime_seconds=round(uptime_seconds, 1) if uptime_seconds else None,
            health_check_url=health_url,
            health_ok=health_ok
        )

    def request_shutdown(self) -> None:
        """Request graceful shutdown (called by signal handlers)."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        logger.info(f"Shutdown requested for {self.service_name}")

        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    # =========================================================================
    # Signal Handlers
    # =========================================================================

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != 'win32':
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
            self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        else:
            # Windows only supports SIGINT
            self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if sys.platform != 'win32':
            if self._original_sigterm:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint:
                signal.signal(signal.SIGINT, self._original_sigint)
        else:
            if self._original_sigint:
                signal.signal(signal.SIGINT, self._original_sigint)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}")
        self.request_shutdown()

    # =========================================================================
    # Private Methods - Process Management
    # =========================================================================

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        if sys.platform == 'win32':
            return self._is_process_running_windows(pid)
        else:
            return self._is_process_running_unix(pid)

    def _is_process_running_windows(self, pid: int) -> bool:
        """Windows-specific process check."""
        try:
            import subprocess
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            return str(pid) in result.stdout
        except Exception:
            return False

    def _is_process_running_unix(self, pid: int) -> bool:
        """Unix-specific process check."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _kill_process(self, pid: int) -> None:
        """Kill a process by PID."""
        if sys.platform == 'win32':
            self._kill_process_windows(pid)
        else:
            self._kill_process_unix(pid)

    def _kill_process_windows(self, pid: int) -> None:
        """Windows-specific process kill with tree kill."""
        try:
            import subprocess
            # Use /T to kill process tree (child processes too)
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            logger.info(f"Killed process tree for PID {pid}")
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")

    def _kill_process_unix(self, pid: int) -> None:
        """Unix-specific process kill."""
        try:
            # First try SIGTERM
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            # If still running, use SIGKILL
            if self._is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
            logger.info(f"Killed process {pid}")
        except OSError as e:
            logger.error(f"Failed to kill process {pid}: {e}")

    def _get_port_process(self) -> Optional[int]:
        """Get the PID of process using our port."""
        if not self.port:
            return None

        if sys.platform == 'win32':
            return self._get_port_process_windows()
        else:
            return self._get_port_process_unix()

    def _get_port_process_windows(self) -> Optional[int]:
        """Windows-specific port process lookup."""
        try:
            import subprocess
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )
            for line in result.stdout.split('\n'):
                if f':{self.port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Failed to get port process: {e}")
        return None

    def _get_port_process_unix(self) -> Optional[int]:
        """Unix-specific port process lookup."""
        try:
            import subprocess
            result = subprocess.run(
                ['lsof', '-i', f':{self.port}', '-t'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except Exception as e:
            logger.warning(f"Failed to get port process: {e}")
        return None

    def _cleanup(self) -> None:
        """Clean up PID file on exit."""
        try:
            if self.pid_file.exists():
                # Only delete if it's our PID
                try:
                    content = self.pid_file.read_text().strip()
                    pid_data = self._parse_pid_file(content)
                    if pid_data and pid_data.pid == os.getpid():
                        self.pid_file.unlink()
                        logger.info(f"Cleaned up PID file for {self.service_name}")
                except (ValueError, OSError, json.JSONDecodeError):
                    pass
        except Exception as e:
            logger.warning(f"Error cleaning up PID file: {e}")

        self._lock_acquired = False


# =============================================================================
# Convenience Functions
# =============================================================================

def ensure_singleton(
    service_name: str,
    port: Optional[int] = None,
    force: bool = False
) -> ProcessManager:
    """
    Convenience function to ensure only one instance of a service runs.

    Args:
        service_name: Name of the service
        port: Optional port to check
        force: If True, kill existing instance

    Returns:
        ProcessManager instance if lock acquired

    Raises:
        SystemExit if another instance is running and force=False
    """
    pm = ProcessManager(service_name, port)

    if not pm.acquire_lock(force=force):
        status = pm.get_status()
        print(f"\nERROR: {service_name} is already running!")
        print(f"  PID: {status.pid}")
        if port:
            print(f"  Port: {port}")
        print(f"\nOptions:")
        print(f"  1. Stop the existing instance first")
        print(f"  2. Use --force to kill existing and start new")
        sys.exit(1)

    return pm


def get_service_status(service_name: str, port: Optional[int] = None) -> ServiceStatus:
    """Get status of a service without acquiring lock."""
    pm = ProcessManager(service_name, port)
    return pm.get_status()


def stop_service(service_name: str, port: Optional[int] = None, graceful: bool = True) -> bool:
    """
    Stop a running service.

    Args:
        service_name: Name of the service
        port: Optional port
        graceful: If True, attempt graceful shutdown first

    Returns:
        True if service was stopped, False if not running
    """
    pm = ProcessManager(service_name, port)
    is_running, pid = pm.is_already_running()

    if is_running:
        pm.kill_existing(pid, graceful=graceful)
        print(f"Stopped {service_name} (PID {pid})")
        return True
    else:
        print(f"{service_name} is not running")
        return False


def print_status(service_name: str, port: Optional[int] = None, as_json: bool = False) -> None:
    """
    Print status of a service.

    Args:
        service_name: Name of the service
        port: Optional port
        as_json: If True, output as JSON
    """
    status = get_service_status(service_name, port)

    if as_json:
        print(status.to_json())
    else:
        print(f"\n{service_name} Status:")
        print(f"  Running: {status.is_running}")
        print(f"  PID: {status.pid or 'N/A'}")
        if status.port:
            print(f"  Port: {status.port}")
            print(f"  Port in use: {status.port_in_use}")
        if status.start_time:
            print(f"  Started: {status.start_time}")
        if status.uptime_seconds:
            hours, remainder = divmod(status.uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        if status.health_check_url:
            health_status = "OK" if status.health_ok else "FAILED"
            print(f"  Health: {health_status}")
        print(f"  PID File: {status.pid_file}")
