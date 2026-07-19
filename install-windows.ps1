<#
.SYNOPSIS
    LLM Coder - Uncensored Edition - Windows installer.

.DESCRIPTION
    Windows equivalent of install.sh: installs Ollama, detects GPU/RAM to
    recommend a model, pulls it, sets up a Python virtual environment and
    dependencies, and optionally registers a logon Scheduled Task so the
    app (and its Routines) keeps running without a terminal open — the
    closest Windows equivalent of the systemd --user service install.sh
    offers on Linux.

    UNVERIFIED: written without a Windows machine to test against (this
    session only had a Linux box available). Review before relying on it,
    and report back anything that doesn't work so it can be fixed.
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Info    { param($m) Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Success { param($m) Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-WarnMsg { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-ErrMsg  { param($m) Write-Host "[ERR]  $m" -ForegroundColor Red }

Write-Host @"

  ██╗     ██╗     ███╗   ███╗     ██████╗ ██████╗ ██████╗ ███████╗██████╗
  ██║     ██║     ████╗ ████║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
  ██║     ██║     ██╔████╔██║    ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
  ██║     ██║     ██║╚██╔╝██║    ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
  ███████╗███████╗██║ ╚═╝ ██║    ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
  ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
  LLM Coder — Uncensored Edition REV 1.1 (Windows)

"@ -ForegroundColor Cyan

# ── 1. Ollama ──────────────────────────────────────────────────────────────
Write-Info "Checking Ollama..."
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Info "Installing Ollama..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    $installed = $false
    if ($winget) {
        try {
            winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
            $installed = $true
        } catch {
            Write-WarnMsg "winget install failed, falling back to the official installer download..."
        }
    }
    if (-not $installed) {
        $installerPath = Join-Path $env:TEMP "OllamaSetup.exe"
        Write-Info "Downloading the official Ollama installer..."
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installerPath
        Write-Info "Running Ollama installer silently..."
        Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT /NORESTART" -Wait
    }
    # Ollama's installer adds itself to PATH but this process' PATH won't see
    # it until a new shell — refresh from the registry so the rest of this
    # script can find it without asking the user to reopen PowerShell.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-ErrMsg "Ollama installed but 'ollama' isn't on PATH yet in this session — close this window, reopen PowerShell, and re-run this script."
        exit 1
    }
    Write-Success "Ollama installed"
} else {
    Write-Success "Ollama already installed: $(ollama --version)"
}

# ── 2. Start Ollama daemon ───────────────────────────────────────────────────
Write-Info "Checking Ollama daemon..."
$ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Info "Starting Ollama daemon..."
    # The Windows installer normally registers Ollama to start itself at
    # logon (tray app) — this is just a fallback for a fresh install where
    # that hasn't kicked in yet this session.
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}
Write-Success "Ollama daemon running"

# ── 3. Detect hardware and recommend a model ─────────────────────────────────
Write-Info "Detecting hardware..."

$sysRamGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

$gpuVendor = "none"
$gpuName = "(none detected)"
$gpuVramGB = 0

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $gpuVendor = "nvidia"
        $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader) | Select-Object -First 1
        $vramMB = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) | Select-Object -First 1
        $gpuVramGB = [math]::Round([double]$vramMB / 1024)
    } catch { }
} else {
    $videoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.AdapterRAM -gt 0 } | Sort-Object AdapterRAM -Descending
    if ($videoControllers) {
        $gpu = $videoControllers | Select-Object -First 1
        $gpuName = $gpu.Name
        # AdapterRAM is a 32-bit field in WMI and wraps/truncates above ~4GB
        # on many drivers, so this is a best-effort figure only — real VRAM
        # may be higher than what's reported here.
        $gpuVramGB = [math]::Round($gpu.AdapterRAM / 1GB)
        if ($gpuName -match "AMD|Radeon") { $gpuVendor = "amd" }
        elseif ($gpuName -match "Intel")  { $gpuVendor = "intel" }
        else { $gpuVendor = "other" }
    }
}

Write-Info "System RAM: ${sysRamGB} GB"
if ($gpuVendor -ne "none") {
    if ($gpuVramGB -gt 0) {
        Write-Info "GPU: $gpuName ($gpuVendor, ~$gpuVramGB GB VRAM)"
    } else {
        Write-Info "GPU: $gpuName ($gpuVendor, VRAM size undetected — sizing models to system RAM instead)"
    }
} else {
    Write-WarnMsg "No dedicated GPU detected — models will run on CPU (slower)"
}

if ($gpuVramGB -gt 0) { $budgetGB = $gpuVramGB } else { $budgetGB = [math]::Floor($sysRamGB * 0.6) }

if     ($budgetGB -ge 20) { $recommended = "qwen2.5-coder:32b";   $recommendedDesc = "32B — most capable coder, needs ~19GB" }
elseif ($budgetGB -ge 10) { $recommended = "qwen2.5-coder:14b";   $recommendedDesc = "14B — best all-round coder, needs ~9GB" }
elseif ($budgetGB -ge 6)  { $recommended = "qwen2.5-coder:7b";    $recommendedDesc = "7B — fast, needs ~5GB" }
else                       { $recommended = "qwen2.5-coder:1.5b"; $recommendedDesc = "1.5B — lightweight, low-resource fallback" }

Write-Info "Recommended for this hardware: $recommended ($recommendedDesc)"

Write-Host ""
Write-Host "Available models (detected budget: ~${budgetGB} GB):" -ForegroundColor Cyan
Write-Host "  1) $recommended"
Write-Host "     $recommendedDesc — recommended for this machine"
Write-Host "  2) qwen2.5-coder:32b-abliterated             — uncensored weights (needs ~19GB, only if you have it)"
Write-Host "  3) qwen2.5-coder:7b                          — fast, low-resource (~5GB)"
Write-Host "  4) deepseek-coder-v2:16b                     — excellent reasoning + code (~10GB)"
Write-Host "  5) Recommended + qwen2.5-coder:7b fallback"
Write-Host "  6) Skip model download"
Write-Host ""
$modelChoice = Read-Host "Choose models to pull [1]"
if ([string]::IsNullOrWhiteSpace($modelChoice)) { $modelChoice = "1" }

function Pull-Model {
    param($Model)
    Write-Info "Pulling $Model..."
    ollama pull $Model
    if ($LASTEXITCODE -eq 0) { Write-Success "Pulled $Model" } else { Write-WarnMsg "Failed to pull $Model" }
}

$chosenModel = $recommended
switch ($modelChoice) {
    "1" { Pull-Model $recommended }
    "2" { Pull-Model "qwen2.5-coder:32b-abliterated"; $chosenModel = "qwen2.5-coder:32b-abliterated" }
    "3" { Pull-Model "qwen2.5-coder:7b"; $chosenModel = "qwen2.5-coder:7b" }
    "4" { Pull-Model "deepseek-coder-v2:16b"; $chosenModel = "deepseek-coder-v2:16b" }
    "5" { Pull-Model $recommended; if ($recommended -ne "qwen2.5-coder:7b") { Pull-Model "qwen2.5-coder:7b" } }
    "6" { Write-WarnMsg "Skipping model download"; $chosenModel = "" }
    default { Pull-Model $recommended }
}

if ($chosenModel -ne "") {
    $configFile = Join-Path $ScriptDir "config.json"
    if (Test-Path $configFile) {
        try {
            $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
            $cfg | Add-Member -NotePropertyName default_model -NotePropertyValue $chosenModel -Force
            $cfg | ConvertTo-Json -Depth 10 | Set-Content $configFile
            Write-Success "Set default_model to $chosenModel in config.json"
        } catch {
            Write-WarnMsg "Couldn't update config.json default_model: $_"
        }
    }
}

# ── 4. Python virtual environment ─────────────────────────────────────────────
Write-Info "Setting up Python environment..."
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { $pythonCmd = Get-Command py -ErrorAction SilentlyContinue }
if (-not $pythonCmd) {
    Write-ErrMsg "Python not found. Install Python 3.11+ from https://python.org/downloads (check 'Add python.exe to PATH' during install) or 'winget install Python.Python.3.12', then re-run this script."
    exit 1
}

$backendDir = Join-Path $ScriptDir "backend"
Push-Location $backendDir
try {
    $venvDir = Join-Path $backendDir "venv"
    if (-not (Test-Path $venvDir)) {
        & $pythonCmd.Source -m venv venv
        Write-Success "Virtual environment created"
    }
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    & $venvPython -m pip install --quiet --upgrade pip
    & $venvPython -m pip install --quiet -r requirements.txt
    Write-Success "Python dependencies installed"
} finally {
    Pop-Location
}

# ── 5. Optional: logon Scheduled Task ────────────────────────────────────────
# Windows' closest equivalent of the systemd --user service on Linux: keeps
# the app (and Routines) running across logins without a terminal open.
Write-Host ""
$installTask = Read-Host "Register LLM Coder to auto-start at logon (keeps Routines running)? [y/N]"
if ($installTask -match "^[Yy]") {
    $launchScript = Join-Path $ScriptDir "launch-windows.ps1"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launchScript`" -NoBrowser"
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    try {
        Register-ScheduledTask -TaskName "LLM Coder" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
        Write-Success "Registered logon task 'LLM Coder' — will auto-start on next login"
        Write-Info "Start it now with: Start-ScheduledTask -TaskName 'LLM Coder'"
        Write-Info "Remove it with: Unregister-ScheduledTask -TaskName 'LLM Coder' -Confirm:`$false"
    } catch {
        Write-WarnMsg "Could not register the Scheduled Task: $_"
    }
} else {
    Write-Info "Skipping auto-start — run launch-windows.ps1 manually (or the Start Menu shortcut) when you want to use LLM Coder"
}

# ── 6. Done ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  LLM Coder — Uncensored Edition REV 1.1            ║" -ForegroundColor Green
Write-Host "║  Installation complete!                             ║" -ForegroundColor Green
Write-Host "║  Run: launch-windows.ps1 (or the logon task, if set) ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
