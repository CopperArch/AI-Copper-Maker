# LLM Coder — Uncensored Edition v0.2

A completely **uncensored**, locally-run AI coding assistant and agent workspace powered by Ollama. No refusals, no guardrails, no limitations — the model will assist with **any** request without hesitation.

Built by [Browns Entertainment](https://github.com/DgBrown21).

---

## v0.2 — What's New

### 🌍 Hardware-agnostic install
`install.sh` now detects GPU vendor/VRAM and system RAM on whatever machine it's run on and recommends/pulls a model sized to fit, instead of assuming a specific GPU. The RDNA4 ROCm gfx-version workaround only applies when that exact hardware is detected. `docker-compose.yml` no longer requires an Nvidia GPU to boot. An optional systemd `--user` service keeps the app (and Routines) running without a terminal open.

### 📧 Email, 📅 Calendar & 👥 Contacts
IMAP/SMTP via app passwords (Gmail, Outlook/Office 365, Yahoo, iCloud, Fastmail, Zoho, AOL, GMX, or custom). The AI drafts 3 reply options for you to pick from — nothing sends automatically. Calendar events get extracted from emails into a pending-approval queue; approved events push to CalDAV (where supported) and always publish to a universal ICS "subscribe by URL" feed, so even Google Calendar (which requires OAuth, not app passwords) can pick them up. Contacts are harvested automatically from your inbox.

### 🧠 Skills & ⏰ Routines
Skills are reusable playbooks the agent can call on mid-task (`get_skill` tool) — write them by hand or have the AI draft one from a pasted conversation/project (you review before it's saved). Routines let you describe a recurring task in plain English; the AI reflects back what it understood, you Agree/Change/Cancel, then adjust the proposed schedule and confirm. Runs on an in-process scheduler with pause/resume/run-now and history.

### 🗂 Semantic Search
Indexes your project files via Ollama embeddings (`ollama pull nomic-embed-text`) and searches them by meaning, not just filename — plus a `semantic_search` agent tool.

### 🔔 Notifications, 💾 Backup, ✏ In-chat Editor
Desktop notifications (via `notify-send`) for finished routines, new pending calendar events, and new mail. One-click backup/restore (zip export of conversations/skills/routines/calendar data, account passwords redacted). Assistant code blocks get an Edit button next to Copy that opens a CodeMirror panel — apply changes back into the conversation.

### 🚫 Totally Uncensored
The system prompt enforces **zero restrictions**. The model will never say:
- "I cannot" / "I can't" / "I'm not able to"
- "I'm sorry, but" / "As an AI, I cannot"
- "For legal/ethical/safety reasons"

It is **free to assist with any request, any topic, any task** without limitation.

### All Features

- **💬 Chat** — interactive conversation with local models (uncensored), with an in-chat code editor
- **🔍 App Analyzer** — research and plan app development
- **⚡ Project Generator** — describe an app, get a full project scaffold
- **▶ Code Runner** — execute Python/JavaScript in sandboxed environments
- **🤖 Agent** — autonomous multi-tool agent
- **📁 File Browser** — manage project files
- **🌐 Search** — web search (DuckDuckGo) plus semantic search over your own files
- **🎨 Image Generation** — generate images via vision models
- **📦 Model Library** — browse, download, and manage Ollama models
- **📧 Email** — read, AI-drafted replies, never auto-sent
- **📅 Calendar** — CalDAV + universal ICS feed, approval-queued event extraction
- **🧠 Skills** — reusable, agent-invokable playbooks
- **⏰ Routines** — scheduled AI tasks with a conversational setup flow

---

## Quick Start

```bash
./install.sh   # detects your hardware, sets up Ollama + a fitting model, installs deps
./launch.sh    # or install the systemd --user service from install.sh for persistent Routines
```

Then open **http://localhost:8081**

Select a model from the sidebar (e.g. `qwen2.5-coder-32b-uncensored` or `mistral-24b-uncensored` — or whatever `install.sh` recommended for your hardware). For semantic search, also pull an embedding model: `ollama pull nomic-embed-text`.

---

## Stack

- **Backend** — FastAPI + Python 3, port 8081
- **Frontend** — single-page HTML/JS, served by FastAPI
- **LLM** — Ollama with local models (chat, embeddings)
- **Code Execution** — sandboxed temp directories, 30s timeout
- **Web Search** — DuckDuckGo HTML scraper (no API key needed)
- **Image Gen** — Ollama vision models (llava, minicpm-v, moondream)
- **Email/Calendar** — stdlib `imaplib`/`smtplib` + `caldav`/`icalendar`
- **Scheduling** — APScheduler (in-process; needs the backend running to fire)
- **Code Editor** — CodeMirror 6 (ESM, no build step)

## Requirements

- Python 3.11+
- Ollama (with your preferred models; `nomic-embed-text` for semantic search)
