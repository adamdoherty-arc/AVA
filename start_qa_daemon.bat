@echo off
REM Magnus QA Daemon - Runs every 10 minutes automatically
REM Start this once and it will keep running in the background

cd /d "c:\code\MagnusAntiG\Magnus"

echo ============================================================
echo   Magnus QA Daemon Starting
echo   Interval: Every 10 minutes
echo   Press Ctrl+C to stop
echo ============================================================

python ".claude\orchestrator\continuous_qa\qa_runner.py" --daemon --interval 10 --visible
