#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same best-effort RDNA4 detection as install.sh (kept in sync manually since
# these are standalone scripts). See install.sh for the rationale.
detect_rdna4_gpu() {
    if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -qi "gfx1201"; then
        return 0
    fi
    if command -v lspci &>/dev/null && lspci 2>/dev/null | grep -iE "AMD/ATI.*(Radeon (RX )?9070|Navi 4[0-9])" &>/dev/null; then
        return 0
    fi
    return 1
}

# gfx1201 (RDNA4, e.g. RX 9070) GPU discovery crashes under early ROCm builds,
# taking the Wayland compositor down with it — force CPU-only there. Everyone
# else keeps normal GPU acceleration. Override with LLM_CODER_FORCE_CPU=1 or
# LLM_CODER_FORCE_GPU=1 if this heuristic gets it wrong on your machine.
if [[ "${LLM_CODER_FORCE_CPU:-}" == "1" ]]; then
    SAFE_MODE=1
elif [[ "${LLM_CODER_FORCE_GPU:-}" == "1" ]]; then
    SAFE_MODE=0
elif detect_rdna4_gpu; then
    SAFE_MODE=1
else
    SAFE_MODE=0
fi

[[ "$SAFE_MODE" == "1" ]] && export OLLAMA_NUM_GPU=0
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=0

# If Ollama is already running, only restart it when we need CPU-only mode
# and it isn't currently in that mode — otherwise leave a working instance alone.
OLLAMA_PID=$(pgrep -x ollama)
if [ -n "$OLLAMA_PID" ]; then
    running_cpu_only=false
    tr '\0' '\n' < /proc/$OLLAMA_PID/environ 2>/dev/null | grep -q "OLLAMA_NUM_GPU=0" && running_cpu_only=true
    if [[ "$SAFE_MODE" == "1" && "$running_cpu_only" == "false" ]]; then
        echo "Restarting Ollama in CPU-only safe mode (RDNA4 workaround)..."
        pkill -x ollama
        sleep 2
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
    else
        echo "Ollama already running ($([ "$running_cpu_only" == "true" ] && echo "CPU-only" || echo "GPU-accelerated"))."
    fi
else
    if [[ "$SAFE_MODE" == "1" ]]; then
        echo "Starting Ollama (CPU-only safe mode — RDNA4 workaround)..."
    else
        echo "Starting Ollama..."
    fi
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Kill any stale backend before starting fresh
pkill -f "uvicorn main:app" 2>/dev/null && sleep 1

# Activate Python venv and start backend
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
echo "Starting LLM Coder backend on http://localhost:8081"
# Bound to localhost only — this app has no login/auth of its own, and it
# exposes full home-directory read/write plus arbitrary code execution
# (/api/execute, /api/files/write, run_command). 0.0.0.0 would put all of
# that on the LAN with zero authentication, reachable from any other device
# on the same network. Change back to 0.0.0.0 only if you specifically want
# LAN access (e.g. from your phone) and understand that tradeoff.
uvicorn main:app --host 127.0.0.1 --port 8081 --reload &
BACKEND_PID=$!

# Wait for backend to start then open browser
sleep 2
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:8081
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  LLM Coder — Uncensored Edition REV 1               ║"
echo "║  Running at http://localhost:8081                    ║"
echo "║  Chat · Agent · Email · Calendar · Skills · Routines ║"
echo "║  Search (web + semantic) · Files · Code Run · Backup ║"
echo "║  Press Ctrl+C to stop                                ║"
echo "╚══════════════════════════════════════════════════════╝"

trap "kill $BACKEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait $BACKEND_PID
