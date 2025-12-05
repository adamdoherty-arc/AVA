# Magnus QA Watchdog - Auto-Start Setup
# This script adds the QA watchdog to Windows Startup (no admin required)
# The watchdog monitors the daemon and restarts it if it crashes

$ProjectRoot = "c:\code\MagnusAntiG\Magnus"
$StartupFolder = [Environment]::GetFolderPath('Startup')
$ShortcutPath = Join-Path $StartupFolder "MagnusQAWatchdog.lnk"

# Remove old daemon shortcut if it exists
$OldShortcut = Join-Path $StartupFolder "MagnusQADaemon.lnk"
if (Test-Path $OldShortcut) {
    Remove-Item $OldShortcut -Force
    Write-Host "Removed old daemon shortcut" -ForegroundColor Yellow
}

# Create shortcut for watchdog
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "pythonw.exe"
$Shortcut.Arguments = "`"$ProjectRoot\.claude\orchestrator\continuous_qa\watchdog.py`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Magnus QA Watchdog - Monitors and restarts QA daemon"
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Save()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Magnus QA Watchdog Auto-Start Configured!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Shortcut created at: $ShortcutPath"
Write-Host ""
Write-Host "  The QA watchdog will now:"
Write-Host "    - Start automatically when Windows starts"
Write-Host "    - Launch and monitor the QA daemon"
Write-Host "    - Restart daemon if it crashes or hangs"
Write-Host "    - Run QA checks every 10 minutes"
Write-Host "    - Send Telegram alerts for issues"
Write-Host "    - Apply automatic fixes"
Write-Host ""
Write-Host "  Log files:"
Write-Host "    - Watchdog: .claude\orchestrator\continuous_qa\logs\watchdog.log"
Write-Host "    - Daemon: .claude\orchestrator\continuous_qa\logs\daemon.log"
Write-Host ""
Write-Host "  To start it NOW, run:" -ForegroundColor Yellow
Write-Host "    .\start_qa_watchdog.bat" -ForegroundColor Yellow
Write-Host ""
Write-Host "  To stop:" -ForegroundColor Yellow
Write-Host "    python .claude\orchestrator\continuous_qa\watchdog.py --stop" -ForegroundColor Yellow
Write-Host ""
