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
  AI Copper Maker — Uncensored Edition REV 1.1
EOF
echo -e "${NC}"

# ── 0. Docker check ────────────────────────────────────────────────────────────
echo ""
info "Checking for container runtime..."
DOCKER_CMD=""
if command -v docker &>/dev/null; then
    DOCKER_CMD="docker"
elif command -v podman &>/dev/null; then
    DOCKER_CMD="podman"
fi

if [ -n "$DOCKER_CMD" ]; then
    echo -e "${CYAN}Installation method:${NC}"
    echo "  1) Native (direct on this machine)"
    echo "  2) Docker/Podman container"
    echo ""
    read -rp "Choose method [1]: " install_method
    install_method="${install_method:-1}"

    if [ "$install_method" = "2" ]; then
        echo ""
        info "Building Docker image with $DOCKER_CMD..."
        cd "$SCRIPT_DIR"
        $DOCKER_CMD compose up -d --build 2>/dev/null || \
        $DOCKER_CMD-compose up -d --build 2>/dev/null || {
            warn "docker-compose not found, building manually..."
            $DOCKER_CMD build -t llm-coder .
            $DOCKER_CMD run -d --name llm-coder-ollama -v ollama-models:/root/.ollama --network host ollama/ollama
            $DOCKER_CMD run -d --name llm-coder-app -p 127.0.0.1:8081:8081 -e OLLAMA_HOST=http://localhost:11434 -v projects:/root/Downloads/LLM-CODER llm-coder
        }
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  AI Copper Maker — Uncensored Edition v0.1.1             ║${NC}"
        echo -e "${GREEN}║  Running in Docker!                                 ║${NC}"
        echo -e "${GREEN}║  Open http://localhost:8081                         ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
        exit 0
    fi
fi

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
    # Ollama's own installer is vendor-maintained, works correctly across every
    # distro (including Arch), and already auto-detects AMD/ROCm GPUs itself —
    # so it's the trustworthy default. An AUR package is unvetted (anyone can
    # publish one) and `--noconfirm` installs it with no chance to review the
    # PKGBUILD first, so AUR is only used as a last-resort fallback if the
    # official installer itself fails, not tried ahead of it.
    curl -fsSL https://ollama.com/install.sh | sh || {
        warn "Official installer failed, trying an AUR helper as a fallback..."
        if command -v yay &>/dev/null; then
            yay -S --noconfirm "$OLLAMA_PKG"
        elif command -v paru &>/dev/null; then
            paru -S --noconfirm "$OLLAMA_PKG"
        else
            error "No AUR helper found either — install Ollama manually from https://ollama.com/download"
        fi
    }
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

# ── 4b. Language Servers (LSP) ──────────────────────────────────────────────────
# Best-effort, every one of these independently skippable: the backend's LSP
# panel (Models tab) just shows whichever of these it can't find as "missing"
# and everything else about the app still works fine — so nothing here is
# allowed to fail the overall install (each step is its own if/&&/|| chain,
# which is safe under this script's `set -e`; a bare failing command
# wouldn't be).
echo ""
info "Installing Language Servers (LSP) for inline diagnostics..."

# rust-analyzer — only if a Rust toolchain (rustup) is already present; this
# script installs Ollama and Python deps, not a whole Rust toolchain just for
# one LSP server nobody asked for otherwise.
if command -v rust-analyzer &>/dev/null; then
    success "rust-analyzer already installed"
elif command -v rustup &>/dev/null; then
    rustup component add rust-analyzer &>/dev/null && success "rust-analyzer installed" \
        || warn "Could not install rust-analyzer via rustup — skipping"
else
    info "Skipping rust-analyzer — no Rust toolchain (rustup) found"
fi

# pylsp (python-lsp-server) — deliberately installed with --user via the
# SYSTEM pip, not this app's own backend/venv above: its console script needs
# to land on ~/.local/bin, which the systemd service's PATH (see the service
# unit below) actually includes — a copy inside backend/venv would be
# invisible to it.
if command -v pylsp &>/dev/null; then
    success "pylsp already installed"
elif command -v pip3 &>/dev/null; then
    pip3 install --quiet --user python-lsp-server && success "pylsp installed" \
        || warn "Could not install pylsp — Python LSP support will show as missing"
else
    warn "No system pip3 found — skipping pylsp"
fi

# clangd (C/C++) comes from the OS package manager, not a per-language
# installer — best-effort across the package managers this script's other
# platform checks already assume exist. Needs sudo; since this script is run
# interactively by hand, that's a normal password prompt here (unlike an
# automated agent trying to sudo non-interactively).
if command -v clangd &>/dev/null; then
    success "clangd already installed"
elif command -v dnf &>/dev/null; then
    sudo dnf install -y clang-tools-extra && success "clangd installed" \
        || warn "Could not install clangd (clang-tools-extra) — you can install it manually with sudo later"
elif command -v apt-get &>/dev/null; then
    sudo apt-get install -y clangd && success "clangd installed" \
        || warn "Could not install clangd — you can install it manually with sudo later"
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm clang && success "clangd installed" \
        || warn "Could not install clangd — you can install it manually with sudo later"
elif command -v brew &>/dev/null; then
    brew install llvm && success "clangd installed (via llvm)" \
        || warn "Could not install clangd via Homebrew"
else
    info "Skipping clangd — no supported package manager detected (dnf/apt/pacman/brew). Install clang-tools-extra (or equivalent) manually for C/C++ LSP support."
fi

# typescript-language-server / bash-language-server run via `npx -y ...` on
# first use — nothing to pre-install; npx fetches them itself the first time
# a .ts/.js/.sh file is opened with LSP enabled.
if command -v npx &>/dev/null; then
    success "npx found — typescript/bash language servers will fetch automatically on first use"
else
    warn "npx not found — typescript/bash LSP support needs Node.js/npm installed"
fi

# ── 5. Optional: auto-start service ─────────────────────────────────────────────
# Running as a service (rather than launching launch.sh by hand) means the app
# survives logout/login and Routines actually fire on schedule instead of only
# while a terminal happens to be open. Linux uses a systemd --user unit; macOS
# has no systemd, so it gets a launchd LaunchAgent doing the same job.
echo ""
if [ "$(uname -s)" = "Darwin" ]; then
    read -rp "Install AI Copper Maker as a launchd agent (auto-start, keeps Routines running)? [y/N]: " install_service
    if [[ "$install_service" =~ ^[Yy]$ ]]; then
        PLIST_LABEL="com.llmcoder.app"
        PLIST_DIR="$HOME/Library/LaunchAgents"
        PLIST_PATH="$PLIST_DIR/$PLIST_LABEL.plist"
        LOG_DIR="$HOME/Library/Logs/LLMCoder"
        mkdir -p "$PLIST_DIR" "$LOG_DIR"
        cat > "$PLIST_PATH" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$PLIST_LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$SCRIPT_DIR/backend/venv/bin/uvicorn</string>
		<string>main:app</string>
		<string>--host</string>
		<string>127.0.0.1</string>
		<string>--port</string>
		<string>8081</string>
	</array>
	<key>WorkingDirectory</key>
	<string>$SCRIPT_DIR/backend</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>StandardOutPath</key>
	<string>$LOG_DIR/stdout.log</string>
	<key>StandardErrorPath</key>
	<string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
PLISTEOF
        # launchctl load/unload are deprecated since 10.11 but still the most
        # broadly-compatible way to (re)register an agent across macOS versions;
        # bootstrap is the modern replacement, tried second on systems where the
        # legacy path fails (e.g. because the label is already bootstrapped).
        launchctl unload "$PLIST_PATH" &>/dev/null || true
        if launchctl load -w "$PLIST_PATH" 2>/dev/null || launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null; then
            success "Installed and started $PLIST_LABEL — will auto-start on login"
            info "Manage it with: launchctl {stop,start} $PLIST_LABEL"
            info "View logs at: $LOG_DIR/stdout.log and stderr.log"
        else
            warn "Could not load the launch agent — try manually: launchctl load -w $PLIST_PATH"
        fi
    else
        info "Skipping launch agent — run ./launch.sh manually when you want to use LLM Coder"
    fi
else
    read -rp "Install AI Copper Maker as a systemd user service (auto-start, keeps Routines running)? [y/N]: " install_service
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
# systemd --user units otherwise get a bare PATH (/usr/local/bin:/usr/bin) —
# no ~/.cargo/bin or ~/.local/bin, which is where rust-analyzer and pylsp
# live, so the LSP status panel would show both "missing" under the service
# even with both actually installed (confirmed live: npx-based servers
# still worked since /usr/bin/npx is on the bare PATH, only the two direct
# binaries were affected).
Environment=PATH=%h/.cargo/bin:%h/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$SCRIPT_DIR/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8081
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
fi

# ── 6. Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  AI Copper Maker — Uncensored Edition REV 1.1            ║${NC}"
echo -e "${GREEN}║  Installation complete!                             ║${NC}"
echo -e "${GREEN}║  Run: ./launch.sh (or the auto-start service, if set up)║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
