@echo off
REM ============================================================================
REM Xtrades Sync Service - Batch Runner
REM ============================================================================
REM This batch file runs the Xtrades sync service for Windows Task Scheduler
REM Schedule: Every 5 minutes during market hours
REM ============================================================================

REM Change to project directory
cd /d "c:\code\MagnusAntiG\Magnus"

REM Activate virtual environment and run sync
call venv\Scripts\activate.bat

REM Run the sync service (output appended to daily log file)
python scripts\utilities\xtrades_sync_service.py >> logs\xtrades_sync.log 2>&1

REM Exit with the Python script's exit code
exit /b %ERRORLEVEL%
