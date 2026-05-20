A web-based code generation and execution tool powered by local LLMs via Ollama.

## Features

- **Chat** — interactive conversation with local coding models
- **App Analyzer** — paste in code, get AI-powered analysis and suggestions
- **Project Generator** — describe an app in natural language, generate a full project
- **Code Runner** — execute Python and JavaScript in sandboxed environments with streaming output
- **Model Library** — browse and manage available Ollama models

## Stack

- **Backend** — FastAPI + Python 3, port 8081
- **Frontend** — single-page HTML/JS, served by FastAPI
- **LLM** — Ollama with local models (default: `qwen2.5-coder:7b`)
- **Code Execution** — sandboxed temp directories, 30s timeout, dangerous patterns blocked

## Quick Start

```bash
./install.sh    # creates venv, installs deps, pulls model
./launch.sh     # starts Ollama + backend
```

Then open http://localhost:8081

## Requirements

- Python 3.11+
- Ollama (CPU-only mode on RDNA4 GPUs)
