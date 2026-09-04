# AI Copper Maker (formerly LLM Coder) REV 1.1 (v1.1)

<p align="center"><img src="brand/logo.png" alt="AI Copper Maker" width="640"></p>

A completely **uncensored**, locally-run AI coding assistant and agent workspace powered by Ollama (with optional LM Studio and cloud-provider models). No refusals, no guardrails, no limitations — the model will assist with **any** request without hesitation. Includes a full mobile app development studio: analyze an app concept and plan its feature set (optionally grounded in a real decompiled `.apk` for interoperability/prototyping), generate a complete project from that feature list, run it live in-browser, and build a real, installable Android APK — entirely offline, without leaving the app.

Built by [CopperArch](https://github.com/DgBrown21).

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) — free to use, modify, and share for any noncommercial purpose. Commercial use (selling it, running it as a paid service, bundling it into a paid product, etc.) requires a separate commercial license — open an issue on this repository to arrange one.

---

## REV 1.1 (v1.1) — What's New

### 🔧 Fully offline Android APK builds
Build Android now offers a **Local (Debug)**, **Local (Release)**, or **EAS Cloud** option. Local builds run entirely on your own machine via the Android SDK + Gradle — no Expo account, no cloud service, nothing leaves your computer. Release builds are signed with an automatically-generated, persistent local keystore, so they're installable straight away. The one-time SDK/JDK setup is auto-detected; the app tells you which build methods are actually available on your machine.

---

## REV 1 (v1.0) — What's New

### 🔁 App Analyzer Clone Mode
A toggle that switches the App Analyzer from "plan a competing app" to "replicate an existing app with full feature parity" — exhaustively inventories the real app's features instead of proposing a cut-down MVP, then hands the feature list straight to the Project Generator with one click.

### 📦 APK Analysis
Drop an `.apk` (or scan your drives — home folder and mounted external drives — for one) to ground Clone Mode in a real decompiled app: package/version/permissions from the manifest, plus a filtered sample of real in-app UI strings pulled from the compiled bytecode, all fed into the analysis as ground truth alongside the model's own knowledge.

### ▶ Run Code for Generated Projects
Generate Project now has a **Run Code** button: saves the project, `npm install`s it, guarantees the web-support packages a generated app routinely omits (react-dom, react-native-web, Metro/webpack runtime depending on Expo SDK age), and starts a live browser preview — embedded directly in the panel via an iframe, with a Stop button. Includes an automatic dependency-repair loop: local models occasionally invent a package version (or a whole package) that was never published, so a failed install is parsed for npm's specific error and retried with the offending entry relaxed or removed, rather than failing the whole run over one bad line the model wrote.

### 📱 Android APK Export
Build a real, installable `.apk` via EAS Build (Expo's free-tier cloud build service) — no local Android SDK needed. Paste a free Expo access token once (validated live against the real account), and it configures, builds, and downloads the finished APK straight into the project folder.

### 📂 Load Saved Project
Reload any previously saved project back into the Generate tab — Save to Disk, Run Code, and Build APK all work on it exactly like a freshly generated one.

### 🐛 Fixes worth knowing about
- The App Analyzer/Project Generator's real token usage wasn't being added to the weekly usage counter at all; the Agent tab wasn't tracking tokens either, and the Chat tab was silently double-counting them. All four are now correct.
- Chat's date/time questions could get answered with a hallucinated date from the model's training cutoff instead of the real one — pure date/time questions are now answered directly from the system clock, and the chat system prompt always carries the real current date/time as grounding for everything else.
- `/api/save-project`'s fallback parser could silently "succeed" while writing garbage filenames and dropping every real file, when a model wrapped its file markers in markdown heading/bold decoration instead of the exact instructed format — now parsed correctly, with fallback pattern ordering fixed so a loose heuristic can't out-run a precise one.
- Docker installs baked a personal `config.json` (save-dir path + a per-install calendar-feed secret) into the image, and only the Ollama service — not the app's own port — was bound to localhost-only in `docker-compose.yml`. Both fixed.
- Update check/apply showed a misleading "could not reach GitHub" on installs with no `.git` directory (e.g. Docker); it now says so plainly instead.
- Generated apps now carry a small "Powered by LLM CODER" credit — requested in the generation prompt, and always guaranteed in the project's README regardless of whether the model complied.

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
- **🔍 App Analyzer** — research and plan app development, or Clone Mode for full feature-parity replication of a real app (optionally grounded in a decompiled `.apk`)
- **⚡ Project Generator** — describe an app, get a full project scaffold; Load a previously saved one back in
- **▶ Run Code** — execute Python/JavaScript in a sandbox, or run a generated project live with an embedded browser preview
- **📱 Build Android APK** — export a generated project as a real installable `.apk` via EAS Build
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

**Linux / macOS:**

```bash
./install.sh   # detects your hardware, sets up Ollama + a fitting model, installs deps
./launch.sh    # or install the systemd --user service from install.sh for persistent Routines
```

**Windows:**

Download and run `AI-Copper-Maker-Setup.exe` (built from `windows-installer/`, see below) — it installs to your user profile (no admin rights needed), adds Start Menu/Desktop shortcuts, and offers to run first-time setup (installs Ollama, downloads a model, sets up the Python environment) at the end.

Or run the scripts directly from a checkout:

```powershell
.\install-windows.ps1   # detects your hardware, sets up Ollama + a fitting model, installs deps
.\launch-windows.ps1    # or register the logon Scheduled Task from install-windows.ps1 for persistent Routines
```

Then open **http://localhost:8081**

Select a model from the sidebar (e.g. `qwen2.5-coder-32b-uncensored` or `mistral-24b-uncensored` — or whatever the installer recommended for your hardware). For semantic search, also pull an embedding model: `ollama pull nomic-embed-text`.

### Building the Windows installer

```bash
windows-installer/build.sh   # needs makensis (`brew install makensis`); outputs windows-installer/dist/AI-Copper-Maker-Setup.exe
```

The build stages only `git`-tracked files plus the Windows launcher scripts, so no local config, account credentials, or signing keystore ever end up in the package.

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
