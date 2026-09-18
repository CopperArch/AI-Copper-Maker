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
LLAMA_SPLIT_MODE=""
CONTEXT_SIZE=4096
N_THREADS=$(nproc 2>/dev/null || echo 4)
BACKEND_LABEL="CPU"
N_GPU_LAYERS=0

# ── Hardware Detection ──────────────────────────────────────
detect_hardware() {
    echo "Detecting hardware..."

    # RAM (in GB)
    TOTAL_RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
    echo "  RAM: ${TOTAL_RAM_GB}GB"

    # GPU detection
    if command -v rocm-smi &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="ROCm"
        VRAM_GB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oP 'Total:\s*\K[0-9]+' | head -1 || echo "0")
        if [ "$VRAM_GB" -gt 0 ] 2>/dev/null; then
            echo "  GPU: ROCm (${VRAM_GB}MB VRAM)"
        else
            echo "  GPU: ROCm detected"
        fi
        # llama.cpp builds with HIP/ROCm support when installed from Fedora repos
        N_GPU_LAYERS=999
        LLAMA_SPLIT_MODE="--split-mode layer"
    elif command -v nvidia-smi &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="CUDA"
        VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        VRAM_GB=$((VRAM_GB / 1024))  # Convert MB to GB
        echo "  GPU: NVIDIA CUDA (${VRAM_GB}GB VRAM)"
        N_GPU_LAYERS=999
        LLAMA_SPLIT_MODE="--split-mode layer"
    elif command -v vulkaninfo &>/dev/null 2>/dev/null; then
        BACKEND_LABEL="Vulkan"
        echo "  GPU: Vulkan detected"
        N_GPU_LAYERS=999
        LLAMA_SPLIT_MODE="--split-mode layer"
    elif [ "$(uname -s)" = "Darwin" ]; then
        BACKEND_LABEL="Metal"
        echo "  GPU: Apple Metal detected"
        N_GPU_LAYERS=999
        LLAMA_SPLIT_MODE="--split-mode layer"
    else
        echo "  GPU: None detected, using CPU"
        N_GPU_LAYERS=0
        LLAMA_SPLIT_MODE=""
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

# ── Select model based on hardware ───────────────────────────
select_model() {
    if [ "$BACKEND_LABEL" = "CPU" ]; then
        # Small model for CPU-only systems
        MODEL_REPO="QuantFactory/qwen2.5-0.5b-instruct-GGUF"
        MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    else
        # Use a 7B model capable of leveraging GPU/ROCm
        MODEL_REPO="QuantFactory/qwen2.5-7b-instruct-GGUF"
        MODEL_FILE="qwen2.5-7b-instruct-Q4_K_M.gguf"
        MODEL_NAME="qwen2.5-7b-instruct-Q4_K_M.gguf"
    fi
    MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
    echo "Selected model: ${MODEL_NAME} (${BACKEND_LABEL})"
}

# ── Install llama.cpp ───────────────────────────────────────
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

# ── Download model ──────────────────────────────────────────────
download_model() {
    if [ -n "$(ls -A "$MODELS_DIR"/*.gguf 2>/dev/null)" ]; then
        echo "Model already exists in $MODELS_DIR."
        return 0
    fi

    echo "No model found. Downloading ${MODEL_NAME}..."
    mkdir -p "$MODELS_DIR"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$MODELS_DIR/$MODEL_NAME" "$MODEL_URL" || {
            echo "Download failed. Place a .gguf file in $MODELS_DIR and re-run."
            exit 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$MODELS_DIR/$MODEL_NAME" "$MODEL_URL" || {
            echo "Download failed. Place a .gguf file in $MODELS_DIR and re-run."
            exit 1
        }
    else
        echo "Neither wget nor curl available. Place a .gguf file in $MODELS_DIR and re-run."
        exit 1
    fi

    echo "Model downloaded to $MODELS_DIR/$MODEL_NAME"
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

# ── Build llama-server flags ────────────────────────────────────
build_flags() {
    local flags=""
    if [ "$N_GPU_LAYERS" -gt 0 ]; then
        flags="--gpu-layers ${N_GPU_LAYERS}"
    fi
    if [ -n "$LLAMA_SPLIT_MODE" ]; then
        flags="${flags} ${LLAMA_SPLIT_MODE}"
    fi
    echo "$flags"
}

# ── Test llama-server ───────────────────────────────────────
test_server() {
    echo "Testing llama-server..."
    local flags
    flags=$(build_flags)
    $LLAMA_SERVER --model "$MODEL_PATH" --host 127.0.0.1 --port 8080 --ctx-size "$CONTEXT_SIZE" --threads "$N_THREADS" $flags &
    LLAMA_PID=$!
    sleep 5

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

# ── Create systemd service ──────────────────────────────────
create_service() {
    mkdir -p "$SERVICE_DIR"
    local flags
    flags=$(build_flags)

    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=llama.cpp OpenAI-compatible API server (${BACKEND_LABEL})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$LLAMA_SERVER --model $MODEL_PATH --host 0.0.0.0 --port 8080 --ctx-size $CONTEXT_SIZE --threads $N_THREADS ${flags}
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
    echo "  GPU layers: ${N_GPU_LAYERS}"
    echo "  Model: $(basename "$MODEL_PATH")"
    echo "  API: http://localhost:8080/v1/models"
    echo "  Chat: http://localhost:8080/v1/chat/completions"
    echo "  Service: systemctl --user status llama-cpp.service"
}

# ── Main ────────────────────────────────────────────────────
detect_hardware
select_model
install_llama_cpp
download_model
find_model
test_server
create_service
