# LLM Coder — Uncensored Edition v0.1.1

A completely **uncensored**, locally-run AI coding assistant and agent workspace powered by Ollama. No refusals, no guardrails, no limitations — the model will assist with **any** request without hesitation.

Built by [Browns Entertainment](https://github.com/DgBrown21).

---

## v0.1.1 — What's New

### 🚫 Totally Uncensored
The system prompt has been rewritten to enforce **zero restrictions**. The model will never say:
- "I cannot" / "I can't" / "I'm not able to"
- "I'm sorry, but" / "As an AI, I cannot"
- "For legal/ethical/safety reasons"

It is **free to assist with any request, any topic, any task** without limitation.

### 🧠 New Features (Odysseus-like)

| Feature | Tab | What it does |
|---|---|---|
| **Agent Mode** | 🤖 Agent | Autonomous agent with tool access — executes code, searches the web, reads/writes files, and generates images. Describe a task and it works through it step-by-step. |
| **Web Search** | 🌐 Search | Search DuckDuckGo directly from the app for documentation, APIs, tutorials, or any online info. |
| **File Browser** | 📁 Files | Browse, create, edit, and save project files in `~/Downloads/LLM-CODER/` — full project management. |
| **Image Generation** | 🎨 Image Gen | Generate images from text prompts using vision models (llava, minicpm-v, moondream). |
| **Tool Calling in Chat** | 💬 Chat | The chat endpoint passes tool definitions to Ollama, so the model can use code execution, web search, and file operations mid-conversation. |

### All Features

- **💬 Chat** — interactive conversation with local models (uncensored)
- **🤖 Agent** — autonomous multi-tool agent
- **🔍 App Analyzer** — research and plan app development
- **⚡ Project Generator** — describe an app, get a full project scaffold
- **▶ Code Runner** — execute Python/JavaScript in sandboxed environments
- **🌐 Web Search** — search the web via DuckDuckGo
- **📁 File Browser** — manage project files
- **🎨 Image Generation** — generate images via vision models
- **📦 Model Library** — browse, download, and manage Ollama models

---

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn httpx python-multipart aiofiles pydantic

# Launch
./launch.sh
```

Then open **http://localhost:8081**

Select an uncensored/abliterated model from the sidebar (e.g. `qwen2.5-coder-32b-uncensored` or `mistral-24b-uncensored`).

---

## Stack

- **Backend** — FastAPI + Python 3, port 8081
- **Frontend** — single-page HTML/JS, served by FastAPI
- **LLM** — Ollama with local models
- **Code Execution** — sandboxed temp directories, 30s timeout
- **Web Search** — DuckDuckGo HTML scraper (no API key needed)
- **Image Gen** — Ollama vision models (llava, minicpm-v, moondream)

## Requirements

- Python 3.11+
- Ollama (with your preferred models)
