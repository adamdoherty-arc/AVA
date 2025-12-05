# Setup Windows Task Scheduler for Premium Scanner
# Run as Administrator: powershell -ExecutionPolicy Bypass -File setup_scheduled_tasks.ps1

$TaskName = "Magnus Premium Scanner"
$ScriptPath = "C:\code\MagnusAntiG\Magnus\scripts\run_scheduled_scan.bat"

# Remove existing tasks
Unregister-ScheduledTask -TaskName "$TaskName - Morning" -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "$TaskName - Midday" -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "$TaskName - Afternoon" -Confirm:$false -ErrorAction SilentlyContinue

# Create Morning Task (9:30 AM)
$MorningTrigger = New-ScheduledTaskTrigger -Daily -At "9:30AM"
$MorningAction = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory "C:\code\MagnusAntiG\Magnus"
$MorningSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask -TaskName "$TaskName - Morning" `
    -Trigger $MorningTrigger `
    -Action $MorningAction `
    -Settings $MorningSettings `
    -Description "Scan for premium opportunities at market open" `
    -RunLevel Limited

Write-Host "Created: $TaskName - Morning (9:30 AM)" -ForegroundColor Green

# Create Midday Task (12:00 PM)
$MiddayTrigger = New-ScheduledTaskTrigger -Daily -At "12:00PM"
$MiddayAction = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory "C:\code\MagnusAntiG\Magnus"
$MiddaySettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask -TaskName "$TaskName - Midday" `
    -Trigger $MiddayTrigger `
    -Action $MiddayAction `
    -Settings $MiddaySettings `
    -Description "Scan for premium opportunities at midday" `
    -RunLevel Limited

Write-Host "Created: $TaskName - Midday (12:00 PM)" -ForegroundColor Green

# Create Afternoon Task (4:00 PM)
$AfternoonTrigger = New-ScheduledTaskTrigger -Daily -At "4:00PM"
$AfternoonAction = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory "C:\code\MagnusAntiG\Magnus"
$AfternoonSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask -TaskName "$TaskName - Afternoon" `
    -Trigger $AfternoonTrigger `
    -Action $AfternoonAction `
    -Settings $AfternoonSettings `
    -Description "Scan for premium opportunities at market close" `
    -RunLevel Limited

Write-Host "Created: $TaskName - Afternoon (4:00 PM)" -ForegroundColor Green

Write-Host ""
Write-Host "Scheduled tasks created successfully!" -ForegroundColor Cyan
Write-Host "View tasks: Task Scheduler > Task Scheduler Library > Magnus Premium Scanner" -ForegroundColor Yellow
