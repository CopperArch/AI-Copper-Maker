<#
.SYNOPSIS
    LLM Coder - Uncensored Edition - Windows launcher.

.DESCRIPTION
    Windows equivalent of launch.sh: starts Ollama if it isn't already
    running, starts the FastAPI backend from the venv created by
    install-windows.ps1, and opens the browser to it.

    UNVERIFIED: written without a Windows machine to test against. Review
    before relying on it, and report back anything that doesn't work.

.PARAMETER NoBrowser
    Skip auto-opening the browser — used by the logon Scheduled Task
    (install-windows.ps1) so a background auto-start doesn't pop a browser
    window at every login.
#>
param(
    [switch]$NoBrowser
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Bound to localhost only — this app has no login/auth of its own, and it
# exposes full home-directory read/write plus arbitrary code execution
# (/api/execute, /api/files/write, run_command). 0.0.0.0 would put all of
# that on the LAN with zero authentication, reachable from any other device
# on the same network. Change back to 0.0.0.0 only if you specifically want
# LAN access (e.g. from your phone) and understand that tradeoff.
$Bind = "127.0.0.1"
$Port = 8081

$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "0"

# Start Ollama if it isn't already running.
$ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "Starting Ollama..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    Write-Host "Ollama already running."
}

# Kill any stale backend before starting fresh.
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "uvicorn main:app" } |
    ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
    }
Start-Sleep -Milliseconds 500

$backendDir = Join-Path $ScriptDir "backend"
$venvPython = Join-Path $backendDir "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERR] No venv found at $venvPython — run install-windows.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting LLM Coder backend on http://localhost:$Port"
$backendProc = Start-Process -FilePath $venvPython `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", $Bind, "--port", $Port `
    -WorkingDirectory $backendDir -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2

if (-not $NoBrowser) {
    Start-Process "http://localhost:$Port"
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗"
Write-Host "║  LLM Coder — Uncensored Edition REV 1.1             ║"
Write-Host "║  Running at http://localhost:$Port                    ║"
Write-Host "║  Chat · Agent · Email · Calendar · Skills · Routines ║"
Write-Host "║  Search (web + semantic) · Files · Code Run · Backup ║"
Write-Host "║  Press Ctrl+C to stop                                ║"
Write-Host "╚══════════════════════════════════════════════════════╝"

try {
    Wait-Process -Id $backendProc.Id
} finally {
    if (-not $backendProc.HasExited) {
        Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped."
}
