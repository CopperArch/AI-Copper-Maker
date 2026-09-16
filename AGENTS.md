# AI Copper Maker — dev context

Local LLM coding studio (FastAPI single-file backend + single-file frontend).
- **Run**: `systemctl --user restart coppermaker` → http://localhost:8081 (auto-starts at login)
- **Model server**: `ollama-serve.service` (user install, ~/.local/share/ollama-user). Default model lives in gitignored `config.json`.
- **Deploy**: git push origin master.
- **Auth**: gh CLI (account CopperArch). Never add co-author trailers or AI-assistant references to commits; author = CopperArch <143160503+DgBrown21@users.noreply.github.com>.
- **No secrets in git**: config.json, *_accounts.json, google_oauth.json, api_keys.json, release.keystore are gitignored on purpose.
- **Agent design**: unrestricted by product premise (uncensored, local-only); sudo routes through the human password prompt in the browser — that gate is how root is obtained, not a refusal. Tool results truncate with explicit markers.
