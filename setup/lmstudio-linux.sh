#!/bin/bash
# Headless LM Studio (llmster) setup for Linux.
#
# Installs llmster via the official LM Studio installer, starts the
# daemon + API server once to confirm it works, and registers a
# systemd --user service so it auto-starts at login.
#
# Confirmed working on this machine (Bazzite/atomic Linux).
set -euo pipefail

echo "Installing llmster (LM Studio headless) via the official installer..."
curl -fsSL https://lmstudio.ai/install.sh | bash

export PATH="$HOME/.lmstudio/bin:$PATH"

if ! command -v lms &>/dev/null; then
    echo "lms not found on PATH after install — check ~/.lmstudio/bin exists and try opening a new shell." >&2
    exit 1
fi

echo "Starting the llmster daemon and API server once, to confirm it works..."
lms daemon up
lms server start
echo "Check http://localhost:1234/v1/models — it should list at least one model."

# ── Persistence: systemd --user service (Linux's equivalent of the
# macOS LaunchAgent above) ────────────────────────────────────────────
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"

LMS_BIN="$HOME/.lmstudio/bin/lms"
SERVICE_FILE="$SERVICE_DIR/llmstudio.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=LM Studio (llmster) daemon + API server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$LMS_BIN daemon up
ExecStartPost=/bin/sh -c 'sleep 2 && $LMS_BIN server start'
Restart=on-failure
RestartSec=5
Environment=PATH=$HOME/.lmstudio/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user stop llmstudio.service 2>/dev/null || true
systemctl --user enable --now llmstudio.service

echo "Systemd --user service installed and enabled — llmster will now start automatically at login."
echo "Verify with: systemctl --user status llmstudio.service"
echo "API server at http://localhost:1234/v1/models"
