#!/usr/bin/env python3
"""ACP (Agent Client Protocol) bridge for AI Copper Maker.

Lets Zed's Agent Panel talk to AI Copper Maker as if it were a native ACP
agent. Zed spawns this script over stdio (per the ACP spec); it forwards
each prompt to the already-running AI Copper Maker backend's /api/agent
endpoint and translates its NDJSON event stream into ACP session/update
notifications. AI Copper Maker's own tool-use agent loop, skills, and
model routing all run unchanged behind this — the bridge only translates.

Requires the backend to be running (systemctl --user start coppermaker.service,
or `python3 main.py` from backend/). Point AI_COPPER_MAKER_URL elsewhere if
it's not on the default localhost:8081.

Standalone smoke test (bypasses Zed):
    python3 acp_bridge.py < /dev/null   # should idle waiting for stdio frames

Wired into Zed via ~/.config/zed/settings.json -> agent_servers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any

import httpx
from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
)
from acp.helpers import (
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message_text,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    Implementation,
    ImageContentBlock,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
)

# Zed pipes stdout to the ACP JSON-RPC transport, so all logging must go to
# stderr — anything on stdout would corrupt the protocol stream.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("coppermaker-acp-bridge")

BACKEND_URL = os.environ.get("AI_COPPER_MAKER_URL", "http://localhost:8081")
DEFAULT_MODEL = os.environ.get("ACP_MODEL", "dagbs/qwen2.5-coder-14b-instruct-abliterated")

PromptBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


def _block_text(block: Any) -> str:
    """ACP content blocks arrive as dicts over the wire but as typed models
    when constructed locally — normalize both to plain text. Non-text
    blocks (images, resources) are skipped; AI Copper Maker's /api/agent
    only accepts a plain string message today."""
    if isinstance(block, dict):
        return block.get("text", "")
    return getattr(block, "text", "") or ""


class CopperMakerAgent(Agent):
    """Bridges one Zed Agent Panel session to AI Copper Maker's /api/agent."""

    def __init__(self) -> None:
        self._conn: Client | None = None
        # /api/agent is stateless per call and expects the full history back
        # each turn (same contract the AI Copper Maker web frontend follows)
        # — so the bridge keeps one running conversation list per ACP session.
        self._conversations: dict[str, list[dict]] = {}
        self._client = httpx.AsyncClient(timeout=None)

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name="ai-copper-maker", version="1.3.2"),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid.uuid4().hex
        self._conversations[session_id] = []
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[PromptBlock],
        **kwargs: Any,
    ) -> PromptResponse:
        conn = self._conn
        assert conn is not None, "prompt() called before on_connect()"
        conv = self._conversations.setdefault(session_id, [])

        user_text = "\n".join(t for t in (_block_text(b) for b in prompt) if t).strip()
        if not user_text:
            return PromptResponse(stop_reason="end_turn")

        payload = {
            "model": DEFAULT_MODEL,
            "message": user_text,
            "conversation": conv,
            "session_id": session_id,
        }

        open_tool_calls: dict[str, str] = {}  # AI Copper Maker tool id -> name
        stop_reason: str = "end_turn"

        try:
            async with self._client.stream("POST", f"{BACKEND_URL}/api/agent", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    await self._handle_event(conn, session_id, event, open_tool_calls)
                    if event.get("type") == "done":
                        self._conversations[session_id] = event.get("conversation", conv)
        except httpx.HTTPError as e:
            await conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(
                    f"\n\n[bridge error] Couldn't reach AI Copper Maker at {BACKEND_URL}: {e}\n"
                    "Is `coppermaker.service` running? (`systemctl --user status coppermaker`)\n"
                ),
            )
            stop_reason = "refusal"

        return PromptResponse(stop_reason=stop_reason)

    async def _handle_event(
        self,
        conn: Client,
        session_id: str,
        event: dict,
        open_tool_calls: dict[str, str],
    ) -> None:
        etype = event.get("type")

        if etype == "token":
            await conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(event.get("content", "")),
            )

        elif etype == "tool_use_start":
            tool_id = str(event.get("id"))
            name = event.get("name", "tool")
            open_tool_calls[tool_id] = name
            await conn.session_update(
                session_id=session_id,
                update=start_tool_call(tool_id, name, kind="other", status="in_progress"),
            )

        elif etype == "tool_result":
            # tool_result doesn't carry the id tool_use_start assigned — AI
            # Copper Maker's loop runs one tool at a time, so matching the
            # single open call by name is safe and avoids depending on
            # internal id formats staying stable.
            name = event.get("name", "tool")
            tool_id = next((i for i, n in open_tool_calls.items() if n == name), name)
            result_text = str(event.get("result", ""))[:4000]
            await conn.session_update(
                session_id=session_id,
                update=update_tool_call(
                    tool_id,
                    status="completed",
                    content=[tool_content(text_block(result_text))],
                ),
            )
            open_tool_calls.pop(tool_id, None)

        elif etype == "sudo_required":
            await conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(
                    f"\n\n[AI Copper Maker needs sudo to run `{event.get('command', '')}`. "
                    f"Approve it from the AI Copper Maker web UI ({BACKEND_URL}) — the Zed bridge "
                    "doesn't relay password prompts yet.]\n"
                ),
            )

        elif etype == "error":
            await conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(f"\n\n[error] {event.get('content', '')}\n"),
            )

        # "tool_use_delta" (partial streamed JSON args) and "tool_call"
        # (fully-parsed args, about to run) are intentionally not forwarded —
        # tool_use_start / tool_result already give Zed a clean start/finish
        # pair without exposing the sandbox's raw, sometimes-invalid partial
        # JSON to the UI.

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        # AI Copper Maker's /api/agent has no cancel endpoint today; the
        # in-flight HTTP stream is simply left to finish server-side.
        log.info("cancel requested for session %s (no-op — backend has no cancel endpoint yet)", session_id)


async def main() -> None:
    await run_agent(CopperMakerAgent())


if __name__ == "__main__":
    asyncio.run(main())
