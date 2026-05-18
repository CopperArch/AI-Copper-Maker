#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROCm override for RX 9070 (RDNA 4 / gfx1201)
export HSA_OVERRIDE_GFX_VERSION=11.0.2

# Start Ollama if not running
if ! pgrep -x ollama &>/dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi

# Activate Python venv and start backend
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
echo "Starting LLM Coder backend on http://localhost:8080"
uvicorn main:app --host 0.0.0.0 --port 8080 --reload &
BACKEND_PID=$!

# Wait for backend to start then open browser
sleep 2
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:8080
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  LLM Coder running at http://localhost:8080  ║"
echo "║  Press Ctrl+C to stop                        ║"
echo "╚══════════════════════════════════════════════╝"

trap "kill $BACKEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait $BACKEND_PID
