#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CPU-only mode: gfx1201 (RDNA4) GPU discovery runner crashes under ROCm gfx1101 override,
# triggering a GPU reset that kills the Wayland compositor. Skip GPU entirely.
export OLLAMA_NUM_GPU=0
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=0

# Restart Ollama if it's not running in CPU-only mode (OLLAMA_NUM_GPU=0).
OLLAMA_PID=$(pgrep -x ollama)
if [ -n "$OLLAMA_PID" ]; then
    if ! tr '\0' '\n' < /proc/$OLLAMA_PID/environ 2>/dev/null | grep -q "OLLAMA_NUM_GPU=0"; then
        echo "Restarting Ollama (not in CPU-only mode)..."
        pkill -x ollama
        sleep 2
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
    else
        echo "Ollama already running in CPU-only mode."
    fi
else
    echo "Starting Ollama (CPU-only)..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Kill any stale backend before starting fresh
pkill -f "uvicorn main:app" 2>/dev/null && sleep 1

# Activate Python venv and start backend
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
echo "Starting LLM Coder backend on http://localhost:8081"
uvicorn main:app --host 0.0.0.0 --port 8081 --reload &
BACKEND_PID=$!

# Wait for backend to start then open browser
sleep 2
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:8081
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  LLM Coder — Uncensored Edition                     ║"
echo "║  Running at http://localhost:8081                    ║"
echo "║  Features: Chat, Agent, Code Run, Web Search,       ║"
echo "║  File Browser, Image Gen, App Analyzer, Project Gen ║"
echo "║  Press Ctrl+C to stop                                ║"
echo "╚══════════════════════════════════════════════════════╝"

trap "kill $BACKEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait $BACKEND_PID
