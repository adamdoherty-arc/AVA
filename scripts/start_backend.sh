#!/bin/bash
# AVA Backend Startup Script
# Usage: ./scripts/start_backend.sh [start|stop|restart|status]

PORT=8002
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.backend.pid"
LOG_FILE="$PROJECT_DIR/logs/backend.log"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        lsof -ti:$PORT 2>/dev/null | head -1
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

start() {
    if is_running; then
        echo "✅ Backend already running (PID: $(get_pid))"
        return 0
    fi

    echo "🚀 Starting AVA backend on port $PORT..."

    # Kill any orphan processes on the port
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 1

    # Activate venv and start
    source "$PROJECT_DIR/venv_mac/bin/activate"

    nohup uvicorn backend.main:app --host 0.0.0.0 --port $PORT > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"

    echo "⏳ Waiting for backend to start..."
    sleep 10

    # Verify it started
    if curl -s --max-time 5 "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
        echo "✅ Backend started successfully (PID: $pid)"
        echo "📋 Log file: $LOG_FILE"
    else
        echo "⚠️  Backend may still be initializing..."
        echo "   Check: curl http://localhost:$PORT/api/health"
        echo "   Logs: tail -f $LOG_FILE"
    fi
}

stop() {
    if ! is_running; then
        echo "❌ Backend not running"
        rm -f "$PID_FILE"
        return 0
    fi

    echo "🛑 Stopping backend..."
    local pid=$(get_pid)

    # Graceful shutdown first
    kill "$pid" 2>/dev/null
    sleep 2

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
    fi

    # Kill any remaining processes on port
    lsof -ti:$PORT | xargs kill -9 2>/dev/null

    rm -f "$PID_FILE"
    echo "✅ Backend stopped"
}

restart() {
    stop
    sleep 2
    start
}

status() {
    if is_running; then
        local pid=$(get_pid)
        echo "✅ Backend is running (PID: $pid)"
        echo ""
        echo "Quick health check:"
        curl -s --max-time 5 "http://localhost:$PORT/api/health" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  Status: OK')
except:
    print('  Status: Initializing...')
" || echo "  Status: Not responding yet"
    else
        echo "❌ Backend is not running"
        echo "   Start with: ./scripts/start_backend.sh start"
    fi
}

logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    *)
        echo "AVA Backend Manager"
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
