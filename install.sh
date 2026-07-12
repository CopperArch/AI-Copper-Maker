#!/usr/bin/env bash
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}"
cat <<'EOF'
  ██╗     ██╗     ███╗   ███╗     ██████╗ ██████╗ ██████╗ ███████╗██████╗
  ██║     ██║     ████╗ ████║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
  ██║     ██║     ██╔████╔██║    ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
  ██║     ██║     ██║╚██╔╝██║    ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
  ███████╗███████╗██║ ╚═╝ ██║    ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
  ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
  LLM Coder — Uncensored Edition v0.1.1
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
            $DOCKER_CMD run -d --name llm-coder-app -p 8081:8081 -e OLLAMA_HOST=http://localhost:11434 -v projects:/root/Downloads/LLM-CODER llm-coder
        }
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  LLM Coder — Uncensored Edition v0.1.1             ║${NC}"
        echo -e "${GREEN}║  Running in Docker!                                 ║${NC}"
        echo -e "${GREEN}║  Open http://localhost:8081                         ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
        exit 0
    fi
fi

# ── 1. Ollama ──────────────────────────────────────────────────────────────────
info "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama (with ROCm for RX 9070)..."
    # Try AUR first (CachyOS/Arch), fall back to official installer
    if command -v yay &>/dev/null; then
        yay -S --noconfirm ollama-rocm 2>/dev/null || {
            warn "ollama-rocm AUR failed, trying official installer..."
            curl -fsSL https://ollama.com/install.sh | sh
        }
    elif command -v paru &>/dev/null; then
        paru -S --noconfirm ollama-rocm 2>/dev/null || curl -fsSL https://ollama.com/install.sh | sh
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    success "Ollama installed"
else
    success "Ollama already installed: $(ollama --version)"
fi

# ROCm env for RDNA 4 (gfx1201 — RX 9070)
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.config/fish/config.fish 2>/dev/null && \
   ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc 2>/dev/null; then
    warn "Setting ROCm GPU override for RDNA 4 (RX 9070)..."
    echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.2' >> ~/.bashrc
    [[ -f ~/.config/fish/config.fish ]] && \
        echo 'set -gx HSA_OVERRIDE_GFX_VERSION 11.0.2' >> ~/.config/fish/config.fish
    success "ROCm override set (shell restart needed for full effect)"
fi

# ── 2. Start Ollama daemon ─────────────────────────────────────────────────────
info "Starting Ollama daemon..."
if ! pgrep -x ollama &>/dev/null; then
    export HSA_OVERRIDE_GFX_VERSION=11.0.2
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi
success "Ollama daemon running"

# ── 3. Pull coding models ──────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Available models (RX 9070 has ~16GB VRAM):${NC}"
echo "  1) qwen2.5-coder:14b                        — Best coding model (~9GB)"
echo "  2) deepseek-coder-v2:16b                     — Excellent reasoning + code (~10GB)"
echo "  3) qwen2.5-coder:32b-abliterated             — Uncensored coding (unlisted, ~19GB)"
echo "  4) qwen2.5-coder:7b                          — Fast coding (~5GB)"
echo "  5) All recommended coding (1 + 2 + 4)"
echo "  6) Skip model download"
echo ""
read -rp "Choose models to pull [5]: " model_choice
model_choice="${model_choice:-5}"

pull_model() {
    info "Pulling $1..."
    export HSA_OVERRIDE_GFX_VERSION=11.0.2
    ollama pull "$1" && success "Pulled $1" || warn "Failed to pull $1"
}

case "$model_choice" in
    1) pull_model "qwen2.5-coder:14b" ;;
    2) pull_model "deepseek-coder-v2:16b" ;;
    3) pull_model "qwen2.5-coder:32b-abliterated" ;;
    4) pull_model "qwen2.5-coder:7b" ;;
    5) pull_model "qwen2.5-coder:14b"; pull_model "deepseek-coder-v2:16b"; pull_model "qwen2.5-coder:7b" ;;
    6) warn "Skipping model download" ;;
    *) pull_model "qwen2.5-coder:14b" ;;
esac

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

# ── 5. Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  LLM Coder — Uncensored Edition v0.1.1             ║${NC}"
echo -e "${GREEN}║  Installation complete!                             ║${NC}"
echo -e "${GREEN}║  Run: ./launch.sh                                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
