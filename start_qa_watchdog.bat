@echo off
:: Magnus QA Watchdog Starter
:: Starts the QA watchdog which keeps the daemon running

cd /d "c:\code\MagnusAntiG\Magnus"

echo ============================================================
echo   Magnus QA Watchdog
echo ============================================================
echo.
echo Starting watchdog to monitor QA daemon...
echo The watchdog will:
echo   - Start the QA daemon (runs every 10 minutes)
echo   - Monitor daemon health
echo   - Restart daemon if it crashes or hangs
echo   - Log to .claude/orchestrator/continuous_qa/logs/
echo.

:: Start watchdog with pythonw (no console window)
start "" pythonw ".claude\orchestrator\continuous_qa\watchdog.py"

echo Watchdog started in background.
echo.
echo To check status, view:
echo   .claude\orchestrator\continuous_qa\data\status.json
echo.
echo To view logs:
echo   .claude\orchestrator\continuous_qa\logs\watchdog.log
echo   .claude\orchestrator\continuous_qa\logs\daemon.log
echo.
echo To stop, run:
echo   python .claude\orchestrator\continuous_qa\watchdog.py --stop
echo.
