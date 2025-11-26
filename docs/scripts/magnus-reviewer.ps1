#===============================================================================
# Magnus Project Auto-Reviewer (PowerShell)
# Runs Claude Code every 20 minutes with comprehensive review + running report
#===============================================================================

# Configuration
$ProjectDir = "C:\code\MagnusAntiG\Magnus"
$IntervalMinutes = 20
$ReportFile = "$ProjectDir\REVIEW_REPORT.md"
$LogDir = "$ProjectDir\.claude-reviews"

# Create directories
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Initialize report file
function Initialize-Report {
    if (-not (Test-Path $ReportFile)) {
        $header = @"
# Magnus Project - Automated Review Report

This is a running log of automated code reviews performed by Claude Code.

**Project:** $ProjectDir
**Started:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

---

"@
        $header | Out-File -FilePath $ReportFile -Encoding UTF8
        Write-Host "✅ Created new report file: $ReportFile" -ForegroundColor Green
    }
}

# Run a single review cycle
function Invoke-ReviewCycle {
    param([int]$CycleNumber)
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $dateStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $cycleLog = "$LogDir\cycle_$dateStamp.md"
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "🔍 REVIEW CYCLE #$CycleNumber" -ForegroundColor Green
    Write-Host "   Time: $timestamp" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # Add cycle header to main report
    $cycleHeader = @"

## Review Cycle #$CycleNumber - $timestamp

"@
    $cycleHeader | Out-File -FilePath $ReportFile -Append -Encoding UTF8
    
    # Change to project directory
    Set-Location $ProjectDir
    
    # The review prompt
    $reviewPrompt = @"
You are reviewing the Magnus AI orchestration project. Perform a COMPREHENSIVE and THOROUGH review.

## Your Mission
Review EVERY file in this project systematically. This project contains multiple AI agents and an orchestration layer - be meticulous.

## Review Checklist (Check ALL of these)

### 1. Architecture & Design
- Agent communication patterns
- Orchestration logic correctness
- State management across agents
- Error propagation between agents
- Deadlock or race condition potential

### 2. Security (CRITICAL)
- API key/secret exposure
- Input validation on all entry points
- Injection vulnerabilities (SQL, command, prompt)
- Authentication/authorization gaps
- Data sanitization

### 3. Bug Detection
- Logic errors in agent decision-making
- Edge cases in orchestration flow
- Null/undefined handling
- Promise/async handling issues
- Off-by-one errors, boundary conditions

### 4. Code Quality
- Dead code or unused functions
- DRY violations
- Complexity hotspots
- Naming clarity
- Consistent patterns

### 5. Performance
- Inefficient loops or algorithms
- Memory leaks
- Unnecessary API calls
- Missing caching opportunities
- Blocking operations

### 6. Error Handling
- Unhandled exceptions
- Missing try/catch blocks
- Poor error messages
- Recovery mechanisms

### 7. Configuration & Environment
- Hardcoded values that should be config
- Environment variable handling
- Default value safety

### 8. Testing Gaps
- Untested critical paths
- Missing edge case tests
- Agent interaction tests

## Output Format
For EACH issue found, report:
```
[SEVERITY: CRITICAL|HIGH|MEDIUM|LOW]
FILE: <filepath>
LINE: <line number if applicable>
ISSUE: <clear description>
FIX: <recommended solution>
```

## Summary Section
At the end, provide:
1. Total issues by severity
2. Top 3 most urgent fixes needed
3. Overall project health score (1-10)
4. What was accomplished/checked this cycle

BE THOROUGH. Check EVERY file. Miss nothing.
"@

    Write-Host "Running Claude Code analysis..." -ForegroundColor Yellow
    Write-Host ""
    
    # Run Claude Code with bypass permissions
    $output = & claude --dangerously-skip-permissions --print $reviewPrompt 2>&1
    
    # Display output
    $output | ForEach-Object { Write-Host $_ }
    
    # Save to cycle log
    $output | Out-File -FilePath $cycleLog -Encoding UTF8
    
    # Append to main report
    $reportContent = @"

### Detailed Findings

$($output -join "`n")

---

"@
    $reportContent | Out-File -FilePath $ReportFile -Append -Encoding UTF8
    
    # Count issues
    $criticalCount = ($output | Select-String -Pattern "CRITICAL" -AllMatches).Matches.Count
    $highCount = ($output | Select-String -Pattern "\[.*HIGH.*\]" -AllMatches).Matches.Count
    
    Write-Host ""
    Write-Host "✅ Review cycle #$CycleNumber complete" -ForegroundColor Green
    Write-Host "   📊 Critical issues: $criticalCount" -ForegroundColor $(if ($criticalCount -gt 0) { "Red" } else { "Gray" })
    Write-Host "   📊 High issues: $highCount" -ForegroundColor $(if ($highCount -gt 0) { "Yellow" } else { "Gray" })
    Write-Host "   📁 Cycle log: $cycleLog" -ForegroundColor Gray
    Write-Host "   📋 Main report: $ReportFile" -ForegroundColor Gray
    
    if ($criticalCount -gt 0) {
        Write-Host ""
        Write-Host "⚠️  ALERT: $criticalCount CRITICAL issues found!" -ForegroundColor Red
        
        # Optional: Windows notification
        try {
            [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null
            $balloon = New-Object System.Windows.Forms.NotifyIcon
            $balloon.Icon = [System.Drawing.SystemIcons]::Warning
            $balloon.BalloonTipTitle = "Magnus Review Alert"
            $balloon.BalloonTipText = "$criticalCount CRITICAL issues found in cycle #$CycleNumber"
            $balloon.BalloonTipIcon = "Warning"
            $balloon.Visible = $true
            $balloon.ShowBalloonTip(5000)
        } catch {
            # Notification failed, continue anyway
        }
    }
    
    return @{
        Critical = $criticalCount
        High = $highCount
        LogFile = $cycleLog
    }
}

# Main execution
function Start-AutoReviewer {
    Clear-Host
    
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║         MAGNUS PROJECT AUTO-REVIEWER                          ║" -ForegroundColor Green
    Write-Host "║         Powered by Claude Code                                ║" -ForegroundColor Green
    Write-Host "╠═══════════════════════════════════════════════════════════════╣" -ForegroundColor Green
    Write-Host "║  Project:  $ProjectDir" -ForegroundColor Cyan
    Write-Host "║  Interval: Every $IntervalMinutes minutes" -ForegroundColor Cyan
    Write-Host "║  Report:   $ReportFile" -ForegroundColor Cyan
    Write-Host "║  Mode:     BYPASS PERMISSIONS ENABLED" -ForegroundColor Yellow
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    
    Initialize-Report
    
    $cycle = 1
    $totalCritical = 0
    $totalHigh = 0
    
    while ($true) {
        $result = Invoke-ReviewCycle -CycleNumber $cycle
        
        $totalCritical += $result.Critical
        $totalHigh += $result.High
        
        $nextRun = (Get-Date).AddMinutes($IntervalMinutes)
        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
        Write-Host "📊 Running Totals: $totalCritical critical, $totalHigh high issues found" -ForegroundColor Magenta
        Write-Host "⏰ Next review at: $($nextRun.ToString('HH:mm:ss'))" -ForegroundColor Cyan
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor DarkGray
        Write-Host ""
        
        $cycle++
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
}

# Run the reviewer
Start-AutoReviewer
