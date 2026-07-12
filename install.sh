#!/usr/bin/env bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detects the specific hardware (AMD RDNA4 / gfx1201, e.g. RX 9070) that needs
# a ROCm gfx-version override. Best-effort and conservative: if detection is
# inconclusive we skip the workaround rather than risk misconfiguring someone
# else's GPU (a wrong HSA_OVERRIDE_GFX_VERSION can break ROCm entirely).
detect_rdna4_gpu() {
    if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -qi "gfx1201"; then
        return 0
    fi
    if command -v lspci &>/dev/null && lspci 2>/dev/null | grep -iE "AMD/ATI.*(Radeon (RX )?9070|Navi 4[0-9])" &>/dev/null; then
        return 0
    fi
    return 1
}

echo -e "${CYAN}"
cat <<'EOF'
  ██╗     ██╗     ███╗   ███╗     ██████╗ ██████╗ ██████╗ ███████╗██████╗
  ██║     ██║     ████╗ ████║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
  ██║     ██║     ██╔████╔██║    ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
  ██║     ██║     ██║╚██╔╝██║    ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
  ███████╗███████╗██║ ╚═╝ ██║    ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
  ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
  LLM Coder — Uncensored Edition v0.2
EOF
echo -e "${NC}"

# ── 1. Ollama ──────────────────────────────────────────────────────────────────
info "Checking Ollama..."
if detect_rdna4_gpu; then
    info "Detected AMD RDNA4 GPU (gfx1201-class, e.g. RX 9070)"
    OLLAMA_PKG="ollama-rocm"
else
    OLLAMA_PKG="ollama"
fi

if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    # Try AUR first (Arch-based distros), fall back to official installer
    if command -v yay &>/dev/null; then
        yay -S --noconfirm "$OLLAMA_PKG" 2>/dev/null || {
            warn "$OLLAMA_PKG AUR install failed, trying official installer..."
            curl -fsSL https://ollama.com/install.sh | sh
        }
    elif command -v paru &>/dev/null; then
        paru -S --noconfirm "$OLLAMA_PKG" 2>/dev/null || curl -fsSL https://ollama.com/install.sh | sh
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    success "Ollama installed"
else
    success "Ollama already installed: $(ollama --version)"
fi

# ROCm gfx-version override — only needed on the specific RDNA4 hardware
# where Ollama/ROCm misidentifies the GPU. Applying this on any other GPU
# (or GPU vendor) would misconfigure or break acceleration for that machine.
if detect_rdna4_gpu; then
    if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.config/fish/config.fish 2>/dev/null && \
       ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc 2>/dev/null; then
        warn "Setting ROCm GPU override for RDNA 4..."
        echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.2' >> ~/.bashrc
        [[ -f ~/.config/fish/config.fish ]] && \
            echo 'set -gx HSA_OVERRIDE_GFX_VERSION 11.0.2' >> ~/.config/fish/config.fish
        success "ROCm override set (shell restart needed for full effect)"
    fi
else
    info "No RDNA4 GPU detected — skipping ROCm gfx-version override (not needed on this hardware)"
fi

# ── 2. Start Ollama daemon ─────────────────────────────────────────────────────
info "Starting Ollama daemon..."
if ! pgrep -x ollama &>/dev/null; then
    detect_rdna4_gpu && export HSA_OVERRIDE_GFX_VERSION=11.0.2
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi
success "Ollama daemon running"

# ── 3. Detect hardware and recommend models ────────────────────────────────────
info "Detecting hardware..."

SYS_RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)

GPU_VENDOR="none"
GPU_NAME="(none detected)"
GPU_VRAM_GB=0

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.0f", $1/1024}')
elif [[ -d /sys/class/drm ]] && compgen -G "/sys/class/drm/card*/device/mem_info_vram_total" &>/dev/null; then
    # AMD amdgpu driver exposes VRAM size directly via sysfs — no rocm-smi needed
    GPU_VENDOR="amd"
    vram_bytes=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | sort -n | tail -1)
    [[ -n "$vram_bytes" ]] && GPU_VRAM_GB=$(( vram_bytes / 1024 / 1024 / 1024 ))
    GPU_NAME=$(lspci 2>/dev/null | grep -iE "VGA|Display|3D controller" | grep -i "AMD/ATI" | head -1 | sed -E 's/^[0-9a-f:.]+ [^:]+: //')
elif command -v lspci &>/dev/null && lspci 2>/dev/null | grep -iE "VGA|Display" | grep -qi "AMD/ATI"; then
    GPU_VENDOR="amd"
    GPU_NAME=$(lspci 2>/dev/null | grep -iE "VGA|Display" | grep -i "AMD/ATI" | head -1 | sed -E 's/^[0-9a-f:.]+ [^:]+: //')
elif command -v lspci &>/dev/null && lspci 2>/dev/null | grep -iE "VGA|Display" | grep -qi intel; then
    GPU_VENDOR="intel"
    GPU_NAME=$(lspci 2>/dev/null | grep -iE "VGA|Display" | grep -i intel | head -1 | sed -E 's/^[0-9a-f:.]+ [^:]+: //')
fi

info "System RAM: ${SYS_RAM_GB} GB"
if [[ "$GPU_VENDOR" != "none" ]]; then
    if [[ "$GPU_VRAM_GB" -gt 0 ]]; then
        info "GPU: ${GPU_NAME} (${GPU_VENDOR}, ~${GPU_VRAM_GB} GB VRAM)"
    else
        info "GPU: ${GPU_NAME} (${GPU_VENDOR}, VRAM size undetected — sizing models to system RAM instead)"
    fi
else
    warn "No dedicated GPU detected — models will run on CPU (slower)"
fi

# Usable budget for model weights: dedicated VRAM if we found a real number,
# else a conservative slice of system RAM (leaves headroom for the OS/runner).
if [[ "$GPU_VRAM_GB" -gt 0 ]]; then
    BUDGET_GB=$GPU_VRAM_GB
else
    BUDGET_GB=$(( SYS_RAM_GB * 60 / 100 ))
fi

# Pick the best-fitting model for the detected budget. These are official
# Ollama library tags (same ones the backend's /api/models/catalog serves).
# The app's "uncensored" behavior mainly comes from its system prompt (see
# UNCENSORED_SYSTEM in backend/main.py), not from requiring specially
# abliterated weights, so a stock coder model works well at every tier;
# qwen2.5-coder:32b-abliterated is offered separately as a true-weights
# uncensored option where hardware can actually run it.
if   [[ "$BUDGET_GB" -ge 20 ]]; then RECOMMENDED="qwen2.5-coder:32b";   RECOMMENDED_DESC="32B — most capable coder, needs ~19GB"
elif [[ "$BUDGET_GB" -ge 10 ]]; then RECOMMENDED="qwen2.5-coder:14b";   RECOMMENDED_DESC="14B — best all-round coder, needs ~9GB"
elif [[ "$BUDGET_GB" -ge 6  ]]; then RECOMMENDED="qwen2.5-coder:7b";    RECOMMENDED_DESC="7B — fast, needs ~5GB"
else                                  RECOMMENDED="qwen2.5-coder:1.5b"; RECOMMENDED_DESC="1.5B — lightweight, low-resource fallback"
fi

info "Recommended for this hardware: ${RECOMMENDED} (${RECOMMENDED_DESC})"

echo ""
echo -e "${CYAN}Available models (detected budget: ~${BUDGET_GB} GB):${NC}"
echo "  1) ${RECOMMENDED}"
echo "     ${RECOMMENDED_DESC} — recommended for this machine"
echo "  2) qwen2.5-coder:32b-abliterated             — uncensored weights (needs ~19GB, only if you have it)"
echo "  3) qwen2.5-coder:7b                          — fast, low-resource (~5GB)"
echo "  4) deepseek-coder-v2:16b                     — excellent reasoning + code (~10GB)"
echo "  5) Recommended + qwen2.5-coder:7b fallback"
echo "  6) Skip model download"
echo ""
read -rp "Choose models to pull [1]: " model_choice
model_choice="${model_choice:-1}"

pull_model() {
    info "Pulling $1..."
    detect_rdna4_gpu && export HSA_OVERRIDE_GFX_VERSION=11.0.2
    ollama pull "$1" && success "Pulled $1" || warn "Failed to pull $1"
}

CHOSEN_MODEL="$RECOMMENDED"
case "$model_choice" in
    1) pull_model "$RECOMMENDED" ;;
    2) pull_model "qwen2.5-coder:32b-abliterated"; CHOSEN_MODEL="qwen2.5-coder:32b-abliterated" ;;
    3) pull_model "qwen2.5-coder:7b"; CHOSEN_MODEL="qwen2.5-coder:7b" ;;
    4) pull_model "deepseek-coder-v2:16b"; CHOSEN_MODEL="deepseek-coder-v2:16b" ;;
    5) pull_model "$RECOMMENDED"; [[ "$RECOMMENDED" != "qwen2.5-coder:7b" ]] && pull_model "qwen2.5-coder:7b" ;;
    6) warn "Skipping model download"; CHOSEN_MODEL="" ;;
    *) pull_model "$RECOMMENDED" ;;
esac

# Record what was actually installed so the app's default matches this machine.
if [[ -n "$CHOSEN_MODEL" ]]; then
    CONFIG_FILE="$SCRIPT_DIR/config.json"
    if [[ -f "$CONFIG_FILE" ]] && command -v python3 &>/dev/null; then
        python3 - "$CONFIG_FILE" "$CHOSEN_MODEL" <<'PYEOF'
import json, sys
path, model = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = json.load(f)
cfg["default_model"] = model
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF
        success "Set default_model to ${CHOSEN_MODEL} in config.json"
    fi
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
info "Setting up Python environment..."
cd "$SCRIPT_DIR/backend"

if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    success "Virtual environment created"
fi

source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
success "Python dependencies installed"

# ── 5. Optional: systemd user service ───────────────────────────────────────────
# Running as a service (rather than launching launch.sh by hand) means the app
# survives logout/login and Routines actually fire on schedule instead of only
# while a terminal happens to be open.
echo ""
read -rp "Install LLM Coder as a systemd user service (auto-start, keeps Routines running)? [y/N]: " install_service
if [[ "$install_service" =~ ^[Yy]$ ]]; then
    SERVICE_DIR="$HOME/.config/systemd/user"
    mkdir -p "$SERVICE_DIR"
    cat > "$SERVICE_DIR/llm-coder.service" <<SERVICEEOF
[Unit]
Description=LLM Coder - Uncensored Edition
After=network.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR/backend
ExecStart=$SCRIPT_DIR/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8081
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
SERVICEEOF
    systemctl --user daemon-reload
    if systemctl --user enable --now llm-coder.service; then
        success "Installed and started llm-coder.service — will auto-start on login"
        info "Manage it with: systemctl --user {status,stop,start,restart} llm-coder.service"
        info "View logs with: journalctl --user -u llm-coder.service -f"
    else
        warn "Could not enable the service (is systemd user linger enabled? try: loginctl enable-linger \$USER)"
    fi
else
    info "Skipping systemd service — run ./launch.sh manually when you want to use LLM Coder"
fi

# ── 6. Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  LLM Coder — Uncensored Edition v0.2               ║${NC}"
echo -e "${GREEN}║  Installation complete!                             ║${NC}"
echo -e "${GREEN}║  Run: ./launch.sh (or the systemd service, if set up)║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
