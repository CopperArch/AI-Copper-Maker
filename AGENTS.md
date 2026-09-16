# AI Copper Maker — dev context

Local LLM coding studio (FastAPI single-file backend + single-file frontend).
- **Run**: `systemctl --user restart coppermaker` → http://localhost:8081 (auto-starts at login)
- **Model server**: `ollama-serve.service` (user install, ~/.local/share/ollama-user). Default model lives in gitignored `config.json`.
- **Deploy**: git push origin master.
- **Auth**: gh CLI (account CopperArch). Never add co-author trailers or AI-assistant references to commits; author = CopperArch <143160503+DgBrown21@users.noreply.github.com>.
- **No secrets in git**: config.json, *_accounts.json, google_oauth.json, api_keys.json, release.keystore are gitignored on purpose.
- **Agent design**: unrestricted by product premise (uncensored, local-only); sudo routes through the human password prompt in the browser — that gate is how root is obtained, not a refusal. Tool results truncate with explicit markers.
- **Two copies of the project exist on this machine**: `~/AI-Copper-Maker` (THIS one — git repo, live service, canonical) and `~/Documents/AI-Copper-Maker` (scratch copy the 2026-09-16 session edited first; contents were synced here and are identical as of REV 1.2). Work here going forward.

## Session state — 2026-09-16 (REV 1.2, deployed + committed)

Shipped and verified live (all endpoint/tool tests passed against local Ollama; service restarted and healthy):
- Chat: opencode-style command bar + `/` slash autocomplete (/new /compact /skills /agent /model /clear /stop /help), `/compact` context compaction (`/api/compact`), auto conversation titles (`/api/title`, only retitles default-prefixed titles), auto-learned-skill toasts.
- Agent loop (opencode/Claude-Code patterns): `todo_write` plan tracking (rendered as live checklist), `edit_file` surgical old→new edits with uniqueness guard + near-miss hints + unified diffs (also from `write_file` on overwrite), `grep_files` content regex search, `task` subagent tool (fresh context, one nesting level, no sudo inside). System prompt now mandates plan-first / search-don't-guess / edit-over-rewrite / delegate-heavy-reading / verify-your-work.
- Projects tab: `🤖 Agent Suggestions` button → `/api/code-projects/{id}/agent-suggestions` → one-click "Run in Agent" cards.
- Paid models: Anthropic prompt caching (system + prefix cache_control breakpoints), history slimming (tool results older than last 4 elided for cloud), per-turn estimated cost + monthly budget meter (`/api/cost/summary`, ledger in gitignored `usage_cost.json`, budget in config.json `cloud_budget`, default $20) shown after the token count; budget editable in the token-usage ⋮ menu.
- Bug fixes found while testing: `/api/agent` used to silently drop a bare `message` field; agent loop now sends `options: {temperature: 0.2, num_ctx: 16384}` (default 0.8 temp made tool-calling flip-flop); `_unified_diff` newlines normalized.
- Auto-skill distillation: fires on edit/write/grep/task too, fuzzy near-duplicate dedupe, 120-skill cap (oldest auto evicted first).

Backup of the pre-1.2 files: `.backup-pre-rev1.2/` (gitignored).

Ideas for next session (none started): subagent UI surfacing (show subagent progress inline instead of one tool row), `/plan` preview mode (draft plan → approve → execute), skill review queue for auto-learned skills, budget hard-stop (currently meter only — never blocks), frontend for per-model spend breakdown from `by_model`.
