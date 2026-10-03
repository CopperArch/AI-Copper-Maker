FROM python:3.14-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libnotify-bin \
    && rm -rf /var/lib/apt/lists/*

# ── LSP toolchain — pinned at build time, no runtime downloads ────────────
# LSP servers run inside THIS container (the agent's file edits happen here),
# so every language server it can offer must exist here. Availability is
# therefore a matter of fact, not of what npx could fetch later:
#   * nodejs / npm / clangd — Debian trixie apt packages (OS-pinned)
#   * typescript-language-server / bash-language-server / typescript —
#     npm global installs at exact versions (binaries land in /usr/bin).
#     typescript is pinned to the last classic-JS major (6.x): 7.x is the
#     native rewrite and no longer ships the tsserver layout that
#     typescript-language-server loads, so the server would start and then
#     fail initialize with "Could not find a valid TypeScript installation".
#   * python-lsp-server — pip, pinned in requirements.txt
#   * shellcheck — apt; bash-language-server shells out to it for every
#     diagnostic, so without it the bash server starts but reports nothing
#   * Rust — rustup with the MINIMAL profile (rustc + cargo + std only — no
#     clippy/rustfmt/docs/extra targets) and a pinned toolchain, plus the
#     rust-analyzer component. A standalone rust-analyzer binary is NOT
#     enough: it shells out to the toolchain to fetch a workspace and cannot
#     analyze without it, so "available" would be a lie. Fixed CARGO_HOME /
#     RUSTUP_HOME keep the toolchain in a stable, non-user-specific place
#     that matches the runtime HOME.
RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs npm clangd shellcheck \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --no-fund --no-audit \
        typescript-language-server@6.0.1 \
        typescript@6.0.3 \
        bash-language-server@5.8.1 \
    && rm -rf /root/.npm

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}

RUN curl -fsSL \
        https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
        -o /tmp/rustup-init \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal \
        --default-toolchain 1.99.0 --component rust-analyzer \
    && rm /tmp/rustup-init \
    && rustc --version \
    && rust-analyzer --version

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8081", "--timeout-graceful-shutdown", "10"]
