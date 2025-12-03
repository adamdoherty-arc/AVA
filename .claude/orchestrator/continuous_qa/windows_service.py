#!/usr/bin/env python3
"""
Windows Service for Magnus QA System

Runs the QA system as a Windows service that:
1. Starts automatically on boot
2. Runs every 20 minutes
3. Shows status in system tray (optional)

Installation:
    python windows_service.py install
    python windows_service.py start

Alternative (Task Scheduler):
    See install_scheduled_task() function
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for service
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "service.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# Try to import pywin32 for Windows service support
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
    PYWIN32_AVAILABLE = True
except ImportError:
    PYWIN32_AVAILABLE = False
    logger.warning(
        "pywin32 not installed. Windows service support unavailable. "
        "Install with: pip install pywin32"
    )


class MagnusQAService:
    """
    Windows Service implementation for Magnus QA.

    If pywin32 is not available, this class provides alternative
    scheduling options.
    """

    _svc_name_ = "MagnusQAService"
    _svc_display_name_ = "Magnus QA & Enhancement Service"
    _svc_description_ = (
        "Runs Claude Code every 20 minutes to review and enhance "
        "the Magnus financial trading platform."
    )

    def __init__(self) -> None:
        """Initialize the service."""
        self.running = False
        self.interval_minutes = 20

    def start(self) -> None:
        """Start the service."""
        self.running = True
        logger.info("Magnus QA Service starting...")

        from .qa_runner import QARunner

        runner = QARunner(
            interval_minutes=self.interval_minutes,
            visible=False,
        )

        logger.info(f"Running QA every {self.interval_minutes} minutes")

        try:
            while self.running:
                try:
                    logger.info("Starting QA cycle...")
                    result = runner.run_once()
                    logger.info(f"QA cycle complete: {result}")
                except Exception as e:
                    logger.error(f"QA cycle failed: {e}", exc_info=True)

                # Wait for next cycle
                if self.running:
                    logger.info(f"Sleeping for {self.interval_minutes} minutes...")
                    for _ in range(self.interval_minutes * 60):
                        if not self.running:
                            break
                        time.sleep(1)

        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
        finally:
            logger.info("Magnus QA Service stopped")

    def stop(self) -> None:
        """Stop the service."""
        self.running = False
        logger.info("Magnus QA Service stopping...")


if PYWIN32_AVAILABLE:
    class MagnusQAServiceFramework(win32serviceutil.ServiceFramework):
        """Windows Service Framework implementation."""

        _svc_name_ = MagnusQAService._svc_name_
        _svc_display_name_ = MagnusQAService._svc_display_name_
        _svc_description_ = MagnusQAService._svc_description_

        def __init__(self, args):
            win32serviceutil.ServiceFramework.__init__(self, args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.service = MagnusQAService()

        def SvcStop(self) -> None:
            """Handle service stop."""
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            self.service.stop()

        def SvcDoRun(self) -> None:
            """Run the service."""
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            self.service.start()


def install_scheduled_task():
    """
    Install a Windows Scheduled Task as an alternative to a service.

    This is easier to set up and doesn't require admin privileges.
    """
    import subprocess

    task_name = "MagnusQA"
    python_exe = sys.executable
    script_path = Path(__file__).parent / "qa_runner.py"

    # Create the scheduled task XML
    xml_content = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Magnus QA and Enhancement Service - Runs every 20 minutes</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT20M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
    <BootTrigger>
      <Enabled>true</Enabled>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{python_exe}"</Command>
      <Arguments>"{script_path}" --once</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''

    # Write XML to temp file
    xml_path = SCRIPT_DIR / "task.xml"
    xml_path.write_text(xml_content, encoding='utf-16')

    try:
        # Delete existing task if any
        subprocess.run(
            ["schtasks", "/delete", "/tn", task_name, "/f"],
            capture_output=True,
        )

        # Create new task
        result = subprocess.run(
            ["schtasks", "/create", "/tn", task_name, "/xml", str(xml_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Scheduled task '{task_name}' created successfully!")
            print("The QA system will run every 20 minutes and on boot.")
        else:
            print(f"Failed to create scheduled task: {result.stderr}")

    finally:
        # Clean up XML file
        if xml_path.exists():
            xml_path.unlink()


def create_startup_shortcut():
    """Create a shortcut in the Windows Startup folder."""
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("winshell not installed. Install with: pip install winshell")
        return

    startup_folder = winshell.startup()
    shortcut_path = Path(startup_folder) / "MagnusQA.lnk"

    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = sys.executable
    shortcut.Arguments = f'"{SCRIPT_DIR / "qa_runner.py"}" --daemon'
    shortcut.WorkingDirectory = str(PROJECT_ROOT)
    shortcut.Description = "Magnus QA Service"
    shortcut.save()

    print(f"Startup shortcut created at: {shortcut_path}")


def main():
    """Main entry point for service management."""
    import argparse

    parser = argparse.ArgumentParser(description='Magnus QA Windows Service')
    parser.add_argument('action', nargs='?', default='help',
                        choices=['install', 'remove', 'start', 'stop', 'restart',
                                 'task', 'startup', 'debug', 'help'],
                        help='Service action')

    args = parser.parse_args()

    if args.action == 'help':
        print("""
Magnus QA Windows Service

Usage:
    python windows_service.py [action]

Actions:
    install     Install as Windows Service (requires pywin32 + admin)
    remove      Remove Windows Service
    start       Start the service
    stop        Stop the service
    restart     Restart the service
    task        Create a Windows Scheduled Task (recommended)
    startup     Create a startup folder shortcut
    debug       Run in debug mode (foreground)

Recommended: Use 'task' to create a scheduled task that runs every 20 minutes.
""")
        return

    if args.action == 'task':
        install_scheduled_task()
        return

    if args.action == 'startup':
        create_startup_shortcut()
        return

    if args.action == 'debug':
        print("Running in debug mode (Ctrl+C to stop)...")
        service = MagnusQAService()
        try:
            service.start()
        except KeyboardInterrupt:
            service.stop()
        return

    # Windows Service actions require pywin32
    if not PYWIN32_AVAILABLE:
        print("pywin32 not installed. Use 'task' or 'startup' instead.")
        print("Install pywin32: pip install pywin32")
        return

    if args.action == 'install':
        win32serviceutil.InstallService(
            MagnusQAServiceFramework,
            MagnusQAService._svc_name_,
            MagnusQAService._svc_display_name_,
        )
        print(f"Service '{MagnusQAService._svc_name_}' installed")

    elif args.action == 'remove':
        win32serviceutil.RemoveService(MagnusQAService._svc_name_)
        print(f"Service '{MagnusQAService._svc_name_}' removed")

    elif args.action == 'start':
        win32serviceutil.StartService(MagnusQAService._svc_name_)
        print(f"Service '{MagnusQAService._svc_name_}' started")

    elif args.action == 'stop':
        win32serviceutil.StopService(MagnusQAService._svc_name_)
        print(f"Service '{MagnusQAService._svc_name_}' stopped")

    elif args.action == 'restart':
        win32serviceutil.RestartService(MagnusQAService._svc_name_)
        print(f"Service '{MagnusQAService._svc_name_}' restarted")


if __name__ == '__main__':
    main()
