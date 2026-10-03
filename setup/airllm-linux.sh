#!/bin/bash
# Headless AirLLM setup for Linux.
#
# AirLLM (github.com/lyogavin/airllm, vendored at vendor/airllm at v4.0.0)
# runs 70B-class models on a single GPU by streaming each layer's weights
# disk -> GPU -> back to the meta device around every forward pass, so the
# VRAM footprint is ~one layer instead of the whole model. This script:
#   * auto-detects hardware (GPU vendor, VRAM, RAM — RAM matters as much as
#     VRAM here: layers stream through host memory)
#   * clones the vendored AirLLM source (shallow, pinned at v4.0.0)
#   * builds a dedicated venv (backend/venv-airllm) with torch + the
#     transformers-from-main that AirLLM v4.0.0 needs, kept OUT of the main
#     app's venv on purpose (same isolation as venv-imagegen)
#   * starts the OpenAI-compatible server (backend/airllm_server.py) on
#     port 8082 and registers a systemd --user service for auto-start
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_DIR/vendor/airllm"
VENV_DIR="$REPO_DIR/backend/venv-airllm"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/airllm.service"
AIRLLM_PORT=8082
BACKEND_LABEL="CPU"
DEFAULT_MODEL="${AIRLLM_DEFAULT_MODEL:-}"

# ── Hardware Detection ──────────────────────────────────────
detect_hardware() {
    echo "Detecting hardware..."

    # RAM (in GB)
    TOTAL_RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    echo "  RAM: ${TOTAL_RAM_GB}GB"
    # Layers stream through host memory between disk and GPU; under 32GB of
    # RAM the sweet spot is a 7B model, 32GB+ is where 70B-class becomes
    # practical (weights stream, only one layer + KV cache live on GPU).
    if [ "$TOTAL_RAM_GB" -lt 32 ] 2>/dev/null; then
        echo "  NOTE: <32GB RAM — prefer a 7B-class model (e.g. meta-llama/Llama-3.1-8B-Instruct)."
    fi

    # GPU detection
    if command -v rocm-smi &>/dev/null; then
        BACKEND_LABEL="ROCm"
        VRAM_GB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'Total:\s*\K[0-9]+' | head -1 || echo "0")
        if [ "$VRAM_GB" -gt 0 ] 2>/dev/null; then
            echo "  GPU: ROCm (${VRAM_GB}MB VRAM)"
        else
            echo "  GPU: ROCm detected"
        fi
        TORCH_INSTALL="pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
    elif command -v nvidia-smi &>/dev/null; then
        BACKEND_LABEL="CUDA"
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        VRAM_GB=$(( (VRAM_MB + 1023) / 1024 ))
        echo "  GPU: NVIDIA CUDA (${VRAM_GB}GB VRAM)"
        TORCH_INSTALL="pip install torch --index-url https://download.pytorch.org/whl/cu121"
    else
        echo "  GPU: None detected, using CPU (AirLLM works on CPU but slow —"
        echo "  it becomes a very careful sequential loader, not a speedup)."
        BACKEND_LABEL="CPU"
        TORCH_INSTALL="pip install torch --index-url https://download.pytorch.org/whl/cpu"
    fi

    if [ "$BACKEND_LABEL" != "CPU" ] && [ "${VRAM_GB:-0}" -lt 8 ] 2>/dev/null; then
        echo "  NOTE: <8GB VRAM — AirLLM's whole point is small-VRAM big models,"
        echo "  but keep context short; 7B is the safe default under 8GB."
    fi
}

# ── Clone the vendored AirLLM source (pinned at v4.0.0) ─────
clone_airllm() {
    if [ -d "$VENDOR_DIR/.git" ]; then
        echo "Vendored AirLLM already present at $VENDOR_DIR."
        echo "  (refresh later with: git -C $VENDOR_DIR pull)"
    else
        echo "Cloning AirLLM v4.0.0 (shallow) into $VENDOR_DIR..."
        mkdir -p "$REPO_DIR/vendor"
        git clone --depth=1 --branch v4.0.0 https://github.com/lyogavin/airllm.git "$VENDOR_DIR"
    fi
}

# ── Build the dedicated venv ────────────────────────────────
setup_venv() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "Creating dedicated AirLLM venv at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi
    echo "Installing AirLLM stack (torch + transformers-from-main + deps)..."
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    # AirLLM v4.0.0's release notes pin transformers to GitHub main (the
    # qwen4_exp / Flash-Next in-tree work is not yet on PyPI); accelerate
    # 1.x is what its setup.py requires. Everything else is a floor.
    $TORCH_INSTALL || "$VENV_DIR/bin/pip" install --quiet torch
    "$VENV_DIR/bin/pip" install --quiet \
        "transformers @ git+https://github.com/huggingface/transformers.git" \
        "accelerate>=1.0" \
        "safetensors" \
        "huggingface-hub" \
        "tqdm" \
        "scipy" \
        "sentencepiece" \
        "fastapi" \
        "uvicorn[standard]" \
        "httpx"
    echo "Done installing."
}

# ── Smoke-test the server ───────────────────────────────────
test_server() {
    echo "Testing the AirLLM server..."
    "$VENV_DIR/bin/python" -m uvicorn airllm_server:app \
        --app-dir "$REPO_DIR/backend" \
        --host 127.0.0.1 --port "$AIRLLM_PORT" --log-level warning &
    local pid=$!
    sleep 6
    if curl -s "http://127.0.0.1:$AIRLLM_PORT/health" > /dev/null 2>&1; then
        echo "AirLLM server is running! API at http://127.0.0.1:$AIRLLM_PORT/v1/models"
    else
        echo "AirLLM server failed to start. Check the log above."
        kill "$pid" 2>/dev/null || true
        exit 1
    fi
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# ── Create systemd service ──────────────────────────────────
create_service() {
    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=AirLLM OpenAI-compatible inference server (streamed-layer LLMs, ${BACKEND_LABEL})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR/backend
ExecStart=$VENV_DIR/bin/python -m uvicorn airllm_server:app --host 127.0.0.1 --port $AIRLLM_PORT
Restart=on-failure
RestartSec=5
Environment=PATH=/usr/bin:/usr/local/bin:/bin

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user stop airllm.service 2>/dev/null || true
    systemctl --user enable --now airllm.service

    echo ""
    echo "Done! AirLLM is configured and running."
    echo "  Backend: ${BACKEND_LABEL}"
    echo "  API: http://localhost:${AIRLLM_PORT}/v1/models"
    echo "  Chat: http://localhost:${AIRLLM_PORT}/v1/chat/completions"
    echo "  Service: systemctl --user status airllm.service"
    echo ""
    echo "Next: open AI Copper Maker's Models tab -> AirLLM and load a model"
    echo "(first load downloads + splits the weights, then it is reusable)."
    if [ -z "$DEFAULT_MODEL" ]; then
        echo "  e.g.  meta-llama/Llama-3.1-8B-Instruct   (7B class, ~8GB VRAM)"
        echo "        Qwen/Qwen3-30B-A3B                (MoE, active ~3B)"
        echo "        huihui_ai/meta-llama-3.1-8B-Instruct-abliterated  (uncensored)"
    fi
}

# ── Main ────────────────────────────────────────────────────
detect_hardware
clone_airllm
setup_venv
test_server
create_service
