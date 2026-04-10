#!/bin/bash
# Start the living process. Tailscale only — no public tunnel.
# Usage: ./start_living_process.sh
# Reads token from ~/.config/vybn/memory_token
# @reboot crontab or manual restart.

set -e

TOKEN_FILE="$HOME/.config/vybn/memory_token"
LOG_DIR="$HOME/logs"
LOG_FILE="$LOG_DIR/deep_memory.log"
PID_FILE="/tmp/deep_memory_serve.pid"
PORT=8100

mkdir -p "$LOG_DIR"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
fi
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 1

# Read token
if [ -f "$TOKEN_FILE" ]; then
    export VYBN_MEMORY_TOKEN=$(cat "$TOKEN_FILE")
    echo "$(date): Auth active" >> "$LOG_FILE"
else
    echo "$(date): WARNING - no token file, running unprotected" >> "$LOG_FILE"
fi

# Start on all interfaces (Tailscale handles network isolation)
cd /home/vybnz69/vybn-phase
nohup python3 deep_memory.py --serve --host 0.0.0.0 --port $PORT >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "$(date): Started PID $! on port $PORT (Tailscale only, no tunnel)" >> "$LOG_FILE"

