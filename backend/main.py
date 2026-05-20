import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="LLM Coder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA = "http://localhost:11434"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CONFIG_FILE = Path(__file__).parent.parent / "config.json"
CONVERSATIONS_FILE = Path(__file__).parent.parent / "conversations.json"
DEFAULT_SAVE_DIR = str(Path.home() / "Downloads" / "LLM-CODER")

DEFAULT_SYSTEM = """You are an expert full-stack mobile app developer. You write clean, production-ready code.
When asked to build an app, you:
1. Provide complete, working code — not placeholders
2. Use React Native (Expo) for cross-platform apps unless told otherwise
3. Structure projects properly with separate components, screens, and services
4. Include package.json with all dependencies
5. Write real API integrations, not mocked data"""


# ── Models ─────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str = DEFAULT_SYSTEM
    stream: bool = True

class AnalyzeRequest(BaseModel):
    model: str
    app_name: str
    category: str
    description: str
    reviews: str = ""
    price_target: str = "free with premium"

class PullRequest(BaseModel):
    model: str

class GenerateRequest(BaseModel):
    model: str
    app_name: str
    platform: str = "react-native"
    features: list[str] = []

class SaveProjectRequest(BaseModel):
    app_name: str
    content: str
    save_dir: str = ""

class ExecuteRequest(BaseModel):
    language: str = "javascript"
    code: str


# ── Sandboxed Code Runner ──────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    "os.system(", "subprocess.", "shutil.rmtree(", "shutil.move(",
    "eval(", "exec(", "__import__(",
]

SANDBOX_JS_WRAPPER = """\
const { vm: vmModule } = require('vm');
const vm = new vmModule.Script(code);
"""

@app.get("/api/runtimes")
async def check_runtimes():
    available = []
    for cmds, label in [
        (["node", "--version"], "node"),
        (["python3", "--version"], "python3"),
        (["python", "--version"], "python"),
    ]:
        try:
            r = subprocess.run(cmds, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                available.append({"name": label, "version": r.stdout.strip()})
        except FileNotFoundError:
            pass
    return {"runtimes": available}

@app.post("/api/execute")
async def execute_code(req: ExecuteRequest):
    code = req.code.strip()
    if not code:
        raise HTTPException(422, "No code provided")

    if req.language == "python":
        for pat in BLOCKED_PATTERNS:
            if pat in code:
                raise HTTPException(422, f"Blocked dangerous pattern: '{pat}'")

    cmd = {"javascript": "node", "python": "python3"}.get(req.language, "node")
    suffix = {"javascript": ".js", "python": ".py"}.get(req.language, ".js")

    if shutil.which(cmd) is None:
        raise HTTPException(400, f"Runtime '{cmd}' not found on this system")

    async def stream():
        tmp = tempfile.mkdtemp(prefix="llm-coder-")
        try:
            filepath = os.path.join(tmp, f"code{suffix}")
            with open(filepath, "w") as f:
                f.write(code)

            proc = await asyncio.create_subprocess_exec(
                cmd, filepath,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp,
                env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
            )

            async def pipe_lines(stream, label):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    yield json.dumps({"type": label, "text": line.decode(errors="replace")}) + "\n"

            async for line in pipe_lines(proc.stdout, "stdout"):
                yield line
            async for line in pipe_lines(proc.stderr, "stderr"):
                yield line

            try:
                await asyncio.wait_for(proc.wait(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                yield json.dumps({"type": "error", "text": "Execution timed out (30s)\n"}) + "\n"
                return

            yield json.dumps({"type": "exit", "code": proc.returncode}) + "\n"

        except FileNotFoundError:
            yield json.dumps({"type": "error", "text": f"Runtime '{cmd}' not found\n"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "text": f"{e}\n"}) + "\n"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/system/ram")
async def system_ram():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) == 2:
                info[parts[0].strip()] = int(parts[1].strip().split()[0])
    return {
        "available_gb": round(info.get("MemAvailable", 0) / 1024 / 1024, 1),
        "total_gb": round(info.get("MemTotal", 0) / 1024 / 1024, 1),
    }

@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA}/api/tags")
            return {"status": "ok", "ollama": r.status_code == 200}
    except Exception:
        return {"status": "ok", "ollama": False}


CATALOG = [
    # Coding
    {"name": "qwen2.5-coder:7b",       "desc": "Fast, lightweight coding assistant",         "size_gb": 4.7,  "category": "coding"},
    {"name": "qwen2.5-coder:14b",      "desc": "Best all-round coding model",                "size_gb": 9.0,  "category": "coding"},
    {"name": "qwen2.5-coder:32b",      "desc": "Most capable coder — may need CPU offload", "size_gb": 19.0, "category": "coding"},
    {"name": "deepseek-coder-v2:16b",  "desc": "Excellent reasoning + code generation",      "size_gb": 10.0, "category": "coding"},
    {"name": "deepseek-coder:6.7b",    "desc": "Compact, fast coder from DeepSeek",          "size_gb": 3.8,  "category": "coding"},
    {"name": "devstral:24b",           "desc": "Mistral's dedicated coding model",            "size_gb": 14.0, "category": "coding"},
    {"name": "codellama:13b",          "desc": "Meta's code-focused Llama 2 model",          "size_gb": 7.4,  "category": "coding"},
    {"name": "starcoder2:15b",         "desc": "BigCode's powerful code model",              "size_gb": 9.1,  "category": "coding"},
    {"name": "codegemma:7b",           "desc": "Google's code-optimised Gemma",              "size_gb": 4.9,  "category": "coding"},
    # General
    {"name": "llama3.2:1b",            "desc": "Tiny but surprisingly capable",              "size_gb": 1.3,  "category": "general"},
    {"name": "llama3.2:3b",            "desc": "Small, fast, good for quick tasks",          "size_gb": 2.0,  "category": "general"},
    {"name": "llama3.1:8b",            "desc": "Meta's solid mid-range model",               "size_gb": 4.7,  "category": "general"},
    {"name": "llama3.3:70b",           "desc": "Meta's latest large model",                  "size_gb": 43.0, "category": "general"},
    {"name": "mistral:7b",             "desc": "Fast, efficient European model",              "size_gb": 4.1,  "category": "general"},
    {"name": "mistral-nemo:12b",       "desc": "Mistral × NVIDIA, great reasoning",          "size_gb": 7.1,  "category": "general"},
    {"name": "gemma2:2b",              "desc": "Google's tiny powerhouse",                    "size_gb": 1.6,  "category": "general"},
    {"name": "gemma2:9b",              "desc": "Google Gemma 2 mid-size",                    "size_gb": 5.5,  "category": "general"},
    {"name": "gemma2:27b",             "desc": "Google Gemma 2 large",                        "size_gb": 16.0, "category": "general"},
    {"name": "phi4:14b",               "desc": "Microsoft Phi-4, strong reasoning",          "size_gb": 9.1,  "category": "general"},
    {"name": "phi3.5:3.8b",            "desc": "Microsoft Phi-3.5, fast & smart",            "size_gb": 2.2,  "category": "general"},
    {"name": "qwen2.5:7b",             "desc": "Alibaba Qwen 2.5 general purpose",           "size_gb": 4.7,  "category": "general"},
    {"name": "qwen2.5:14b",            "desc": "Qwen 2.5 mid-size",                          "size_gb": 9.0,  "category": "general"},
    {"name": "qwen2.5:72b",            "desc": "Qwen 2.5 flagship — needs CPU offload",      "size_gb": 47.0, "category": "general"},
    {"name": "deepseek-r1:7b",         "desc": "DeepSeek R1 distilled reasoning model",      "size_gb": 4.7,  "category": "general"},
    {"name": "deepseek-r1:14b",        "desc": "DeepSeek R1 mid-size reasoning",             "size_gb": 9.0,  "category": "general"},
    {"name": "deepseek-r1:32b",        "desc": "DeepSeek R1 large reasoning",                "size_gb": 19.0, "category": "general"},
    # Vision
    {"name": "llava:7b",               "desc": "Visual question answering, 7B",              "size_gb": 4.5,  "category": "vision"},
    {"name": "llava:13b",              "desc": "Visual question answering, 13B",             "size_gb": 8.0,  "category": "vision"},
    {"name": "minicpm-v:8b",           "desc": "Compact but powerful vision model",          "size_gb": 5.5,  "category": "vision"},
    {"name": "moondream",              "desc": "Tiny vision model, great for demos",          "size_gb": 1.7,  "category": "vision"},
    {"name": "llava-phi3",             "desc": "Phi-3 based vision-language model",          "size_gb": 2.9,  "category": "vision"},
]

# ── Models ─────────────────────────────────────────────────────────────────────

@app.get("/api/models/catalog")
async def model_catalog():
    return {"catalog": CATALOG}

@app.get("/api/models")
async def list_models():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{OLLAMA}/api/tags")
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}
        except Exception:
            return {"models": []}

@app.post("/api/models/pull")
async def pull_model(req: PullRequest):
    async def stream():
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream("POST", f"{OLLAMA}/api/pull",
                                     json={"name": req.model}) as r:
                async for line in r.aiter_lines():
                    if line:
                        yield line + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")

@app.delete("/api/models/{model_name:path}")
async def delete_model(model_name: str):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.request("DELETE", f"{OLLAMA}/api/delete",
                                 json={"name": model_name})
        return {"success": r.status_code == 200}


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": req.system}] + \
               [{"role": m.role, "content": m.content} for m in req.messages]

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST", f"{OLLAMA}/api/chat",
                    json={"model": req.model, "messages": messages, "stream": True}
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except httpx.ReadTimeout:
            yield json.dumps({"error": "Ollama timed out — the model may be overloaded"}).encode()
        except Exception as e:
            yield json.dumps({"error": f"Ollama error: {str(e)}"}).encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── App Analyzer ───────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_app(req: AnalyzeRequest):
    prompt = f"""Analyze this app opportunity and create a full development plan:

APP: {req.app_name}
CATEGORY: {req.category}
DESCRIPTION: {req.description}
USER COMPLAINTS/REQUESTS FROM REVIEWS:
{req.reviews or "Not provided"}
PRICING MODEL: {req.price_target}

Provide:

## 1. Market Analysis
- What users hate about existing apps
- Your competitive advantage

## 2. Core Features (MVP)
List the essential features to ship first

## 3. Monetization Strategy
How to price and upsell given the target price

## 4. Technical Architecture
- Platform recommendation (React Native/Flutter/PWA)
- Key libraries and APIs needed
- Backend requirements (if any)

## 5. Complete Project Scaffold
Provide the full file structure and starter code for the main screens.
Include real, working code — not pseudocode.

## 6. App Store Listing
- Suggested app name (unique, searchable)
- Short description (80 chars)
- Full description optimized for ASO
- 5 keyword suggestions"""

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST", f"{OLLAMA}/api/chat",
                    json={
                        "model": req.model,
                        "messages": [
                            {"role": "system", "content": "You are an expert mobile app developer and ASO specialist. Always provide complete, working code."},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield json.dumps({"error": f"Ollama error: {str(e)}"}).encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Project Generator ──────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate_project(req: GenerateRequest):
    features_str = "\n".join(f"- {f}" for f in req.features) if req.features else "- Core app functionality"
    prompt = f"""Generate a complete, ready-to-run {req.platform} project for: {req.app_name}

Required features:
{features_str}

Output the complete project as a series of files. For each file use this format:
=== FILE: path/to/filename.ext ===
[complete file content]
=== END FILE ===

Include:
- package.json with all dependencies
- App.js / App.tsx entry point
- All screen components
- Navigation setup
- Any required API service files
- README.md with setup instructions

Write production-quality code, not demos."""

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream(
                    "POST", f"{OLLAMA}/api/chat",
                    json={
                        "model": req.model,
                        "messages": [
                            {"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": True
                    }
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield json.dumps({"error": f"Ollama error: {str(e)}"}).encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Conversations ──────────────────────────────────────────────────────────────

@app.get("/api/conversations")
async def get_conversations():
    if CONVERSATIONS_FILE.exists():
        try:
            return json.loads(CONVERSATIONS_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []

@app.post("/api/conversations")
async def save_conversations(request: Request):
    try:
        data = await request.json()
        CONVERSATIONS_FILE.write_text(json.dumps(data, indent=2))
        return {"ok": True}
    except (OSError, IOError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to save conversations: {e}")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"save_dir": DEFAULT_SAVE_DIR}

def write_config(cfg: dict):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except (OSError, IOError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")

@app.get("/api/config")
async def get_config():
    return load_config()

@app.post("/api/config")
async def set_config(cfg: dict):
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=422, detail="Expected a JSON object")
    existing = load_config()
    existing.update(cfg)
    write_config(existing)
    return {"ok": True}


# ── Save Project ───────────────────────────────────────────────────────────────

@app.post("/api/save-project")
async def save_project(req: SaveProjectRequest):
    cfg = load_config()
    base = Path(req.save_dir or cfg.get("save_dir", DEFAULT_SAVE_DIR)).expanduser().resolve()

    folder_name = re.sub(r'[^\w\s-]', '', req.app_name).strip().replace(' ', '-')
    project_dir = (base / folder_name).resolve()

    # Parse === FILE: path === ... === END FILE === blocks
    file_pattern = re.compile(
        r'===\s*FILE:\s*(.+?)\s*===\n(.*?)===\s*END FILE\s*===',
        re.DOTALL
    )
    matches = file_pattern.findall(req.content)

    # Fallback: parse markdown fenced code blocks with a filename comment
    if not matches:
        fence_pattern = re.compile(
            r'(?:#+\s*)?`{3,}(?:\w+)?\s*\n(?://|#|<!--)\s*(.+?)\s*(?:-->)?\n(.*?)`{3,}',
            re.DOTALL
        )
        matches = fence_pattern.findall(req.content)

    if not matches:
        raise HTTPException(status_code=422, detail="No parseable file blocks found in output")

    saved = []
    for rel_path, content in matches:
        target = (project_dir / rel_path.strip()).resolve()
        # Safety: ensure we never write outside project_dir
        if not str(target).startswith(str(project_dir)):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.strip() + "\n")
        saved.append(str(target.relative_to(project_dir)))

    return {"saved": saved, "project_dir": str(project_dir), "file_count": len(saved)}


# ── Serve Frontend ─────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
