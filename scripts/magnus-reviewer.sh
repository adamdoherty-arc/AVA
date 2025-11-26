#!/bin/bash
#===============================================================================
# Magnus Project Auto-Reviewer
# Runs Claude Code every 20 minutes with comprehensive review + running report
#===============================================================================

# Configuration
PROJECT_DIR="/c/code/MagnusAntiG/Magnus"  # Git Bash path format
INTERVAL_MINUTES=20
REPORT_FILE="$PROJECT_DIR/REVIEW_REPORT.md"
LOG_DIR="$PROJECT_DIR/.claude-reviews"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR"

# Initialize or update the main report file
init_report() {
    if [ ! -f "$REPORT_FILE" ]; then
        cat > "$REPORT_FILE" << 'EOF'
# Magnus Project - Automated Review Report

This is a running log of automated code reviews performed by Claude Code.

---

EOF
        echo -e "${GREEN}✅ Created new report file: $REPORT_FILE${NC}"
    fi
}

# Run a single review cycle
run_review() {
    local cycle_num=$1
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local date_stamp=$(date +"%Y%m%d_%H%M%S")
    local cycle_log="$LOG_DIR/cycle_${date_stamp}.md"
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🔍 REVIEW CYCLE #$cycle_num${NC}"
    echo -e "${BLUE}   Time: $timestamp${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    
    # Add cycle header to main report
    cat >> "$REPORT_FILE" << EOF

## Review Cycle #$cycle_num - $timestamp

EOF

    cd "$PROJECT_DIR" || { echo "Failed to cd to project"; exit 1; }
    
    # Run Claude Code with bypass permissions and comprehensive review prompt
    echo -e "${YELLOW}Running Claude Code analysis...${NC}"
    
    claude --dangerously-skip-permissions --print "You are reviewing the Magnus AI orchestration project. Perform a COMPREHENSIVE and THOROUGH review.

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
\`\`\`
[SEVERITY: CRITICAL|HIGH|MEDIUM|LOW]
FILE: <filepath>
LINE: <line number if applicable>
ISSUE: <clear description>
FIX: <recommended solution>
\`\`\`

## Summary Section
At the end, provide:
1. Total issues by severity
2. Top 3 most urgent fixes needed
3. Overall project health score (1-10)
4. What was accomplished/checked this cycle

BE THOROUGH. Check EVERY file. Miss nothing." 2>&1 | tee "$cycle_log"

    # Append results to main report
    echo "" >> "$REPORT_FILE"
    echo "### Detailed Findings" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    cat "$cycle_log" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "---" >> "$REPORT_FILE"
    
    # Extract summary stats if possible
    local critical_count=$(grep -c "CRITICAL" "$cycle_log" 2>/dev/null || echo "0")
    local high_count=$(grep -c "HIGH" "$cycle_log" 2>/dev/null || echo "0")
    
    echo ""
    echo -e "${GREEN}✅ Review cycle #$cycle_num complete${NC}"
    echo -e "   📊 Critical issues: ${RED}$critical_count${NC}"
    echo -e "   📊 High issues: ${YELLOW}$high_count${NC}"
    echo -e "   📁 Cycle log: $cycle_log"
    echo -e "   📋 Main report: $REPORT_FILE"
    
    # Alert if critical issues found
    if [ "$critical_count" -gt 0 ]; then
        echo -e "${RED}⚠️  ALERT: $critical_count CRITICAL issues found!${NC}"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║         MAGNUS PROJECT AUTO-REVIEWER                         ║"
    echo "║         Powered by Claude Code                               ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Project: $PROJECT_DIR"
    echo "║  Interval: Every $INTERVAL_MINUTES minutes"
    echo "║  Report: $REPORT_FILE"
    echo "║  Permissions: BYPASSED (--dangerously-skip-permissions)      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    init_report
    
    local cycle=1
    
    while true; do
        run_review $cycle
        
        echo ""
        echo -e "${BLUE}⏰ Next review in $INTERVAL_MINUTES minutes ($(date -d "+$INTERVAL_MINUTES minutes" +"%H:%M:%S"))${NC}"
        echo ""
        
        cycle=$((cycle + 1))
        sleep $((INTERVAL_MINUTES * 60))
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Stopping reviewer... Final report at: $REPORT_FILE${NC}"; exit 0' SIGINT

# Run
main
