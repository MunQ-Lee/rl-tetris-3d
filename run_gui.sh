#!/bin/bash
# Supervisor: keep the GUI server up, restarting it if it ever exits.
# Launch detached:  setsid nohup ./run_gui.sh >/dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
PORT="${1:-8000}"
SPEED="${2:-8}"
while true; do
    echo "[$(date '+%H:%M:%S')] starting server on :$PORT" >> server.log
    python3 server.py --port "$PORT" --speed "$SPEED" >> server.log 2>&1
    echo "[$(date '+%H:%M:%S')] server exited (code $?), restarting in 3s" >> server.log
    sleep 3
done
