@echo off
REM Run Premium Scanner - For Windows Task Scheduler
REM Schedule this to run at 9:30 AM and 4:00 PM

cd /d "C:\code\MagnusAntiG\Magnus"
call venv\Scripts\activate
python scripts/scheduled_premium_scanner.py --once

REM Alternative: Run Robinhood scanner directly with TradingView symbols
REM python scripts/background_premium_scanner_robinhood.py --tradingview
