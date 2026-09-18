#!/bin/bash
# Headless llama.cpp setup for Linux.
#
# Auto-detects hardware (CPU, GPU, RAM, threads), selects the optimal
# llama.cpp backend, downloads a default GGUF model, starts the
# OpenAI-compatible API server on port 8080, and registers a
# systemd --user service for auto-start at login.
set -euo pipefail

LLAMA_SERVER=/usr/bin/llama-server
MODELS_DIR="$HOME/.llama/models"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/llama-cpp.service"
LLAMA_CPP_BACKEND_FLAGS=""
CONTEXT_SIZE=4096
N_THREADS=$(nproc 2>/dev/null || echo 4)
BACKEND_LABEL="CPU"

# ── Hardware Detection ──────────────────────────────────────────────
detect_hardware() {
    echo "Detecting hardware..."

    # RAM (in GB)
    TOTAL_RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    echo "  RAM: ${TOTAL_RAM_GB}GB"

    # GPU detection
    if command -v rocm-smi &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="ROCm"
        # Get VRAM info
        VRAM_GB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'Total:\s*\K[0-9]+' | head -1 || echo "0")
        if [ "$VRAM_GB" -gt 0 ] 2>/dev/null; then
            echo "  GPU: ROCm (${VRAM_GB}MB VRAM)"
        else
            echo "  GPU: ROCm detected"
        fi
        # llama.cpp builds with HIP/ROCm support when installed from Fedora repos
        LLAMA_CPP_BACKEND_FLAGS="--gpu-layers 999"
    elif command -v nvidia-smi &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="CUDA"
        VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        VRAM_GB=$((VRAM_GB / 1024))  # Convert MB to GB
        echo "  GPU: NVIDIA CUDA (${VRAM_GB}GB VRAM)"
        LLAMA_CPP_BACKEND_FLAGS="--gpu-layers 999"
    elif command -v vulkaninfo &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="Vulkan"
        echo "  GPU: Vulkan detected"
        LLAMA_CPP_BACKEND_FLAGS="--gpu-layers 999"
    elif [ "$(uname -s)" = "Darwin" ]; then
        BACKEND_LABEL="Metal"
        echo "  GPU: Apple Metal detected"
        LLAMA_CPP_BACKEND_FLAGS="--gpu-layers 999"
    else
        echo "  GPU: None detected, using CPU"
        LLAMA_CPP_BACKEND_FLAGS=""
    fi

    # Threads: use logical cores but cap at reasonable number
    N_THREADS=$(nproc 2>/dev/null || echo 4)
    echo "  Threads: ${N_THREADS}"

    # Adjust context size based on available RAM
    if [ "$TOTAL_RAM_GB" -ge 32 ] 2>/dev/null; then
        CONTEXT_SIZE=16384
    elif [ "$TOTAL_RAM_GB" -ge 16 ] 2>/dev/null; then
        CONTEXT_SIZE=8192
    elif [ "$TOTAL_RAM_GB" -ge 8 ] 2>/dev/null; then
        CONTEXT_SIZE=4096
    else
        CONTEXT_SIZE=2048
    fi
    echo "  Context size: ${CONTEXT_SIZE}"
    echo "  Backend: ${BACKEND_LABEL}"
}

# ── Install llama.cpp ───────────────────────────────────────────────
install_llama_cpp() {
    if [ ! -x "$LLAMA_SERVER" ]; then
        echo "llama-server not found. Installing llama-cpp via dnf..."
        sudo dnf install -y llama-cpp 2>&1 | tail -5
    fi
    if [ ! -x "$LLAMA_SERVER" ]; then
        echo "ERROR: llama-server not available. Install llama-cpp manually first."
        exit 1
    fi
}

# ── Download model ──────────────────────────────────────────────────
download_model() {
    if [ -n "$(ls -A "$MODELS_DIR"/*.gguf 2>/dev/null)" ]; then
        echo "Model already exists in $MODELS_DIR."
        return 0
    fi

    echo "No model found. Downloading a small chat model..."
    mkdir -p "$MODELS_DIR"

    # Download a small Q4_K_M quantized model suitable for the detected hardware
    local model_url="https://huggingface.co/QuantFactory/qwen2.5-0.5b-instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    local model_name="qwen2.5-0.5b-instruct-q4_k_m.gguf"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$MODELS_DIR/$model_name" "$model_url" 2>/dev/null || {
            echo "Download failed. Place a .gguf file in $MODELS_DIR and re-run."
            exit 1
        }
    elif command -v curl &>/dev/null; then
        curl -L -o "$MODELS_DIR/$model_name" "$model_url" 2>/dev/null || {
            echo "Download failed. Place a .gguf file in $MODELS_DIR and re-run."
            exit 1
        }
    else
        echo "Neither wget nor curl available. Place a .gguf file in $MODELS_DIR and re-run."
        exit 1
    fi

    echo "Model downloaded to $MODELS_DIR/$model_name"
}

# ── Find model path ─────────────────────────────────────────────────
find_model() {
    MODEL_PATH="$(ls -A "$MODELS_DIR"/*.gguf 2>/dev/null | head -1)"
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: No .gguf model found in $MODELS_DIR."
        exit 1
    fi
    echo "Using model: $MODEL_PATH"
}

# ── Test llama-server ───────────────────────────────────────────────
test_server() {
    echo "Testing llama-server..."
    $LLAMA_SERVER --model "$MODEL_PATH" --host 127.0.0.1 --port 8080 --ctx-size "$CONTEXT_SIZE" --threads "$N_THREADS" $LLAMA_CPP_BACKEND_FLAGS &
    LLAMA_PID=$!
    sleep 3

    if curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
        echo "llama-server is running! API at http://127.0.0.1:8080/v1/models"
    else
        echo "llama-server failed to start. Check logs above."
        kill $LLAMA_PID 2>/dev/null || true
        exit 1
    fi
    kill $LLAMA_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
}

# ── Create systemd service ──────────────────────────────────────────
create_service() {
    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=llama.cpp OpenAI-compatible API server (${BACKEND_LABEL})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$LLAMA_SERVER --model $MODEL_PATH --host 0.0.0.0 --port 8080 --ctx-size $CONTEXT_SIZE --threads $N_THREADS $LLAMA_CPP_BACKEND_FLAGS
Restart=on-failure
RestartSec=5
Environment=PATH=/usr/bin:/usr/local/bin:/bin

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user stop llama-cpp.service 2>/dev/null || true
    systemctl --user enable --now llama-cpp.service

    echo ""
    echo "Done! llama.cpp is configured and running."
    echo "  Backend: ${BACKEND_LABEL}"
    echo "  Threads: ${N_THREADS}"
    echo "  Context: ${CONTEXT_SIZE}"
    echo "  Model: $(basename "$MODEL_PATH")"
    echo "  API: http://localhost:8080/v1/models"
    echo "  Chat: http://localhost:8080/v1/chat/completions"
    echo "  Service: systemctl --user status llama-cpp.service"
}

# ── Main ────────────────────────────────────────────────────────────
detect_hardware
install_llama_cpp
download_model
find_model
test_server
create_service
