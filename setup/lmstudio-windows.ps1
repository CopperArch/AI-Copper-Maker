# Headless LM Studio (llmster) setup for Windows.
#
# UNVERIFIED: the install command below is Windows' official one per
# https://lmstudio.ai/docs/developer/core/headless. Everything after that —
# the persistence-at-login setup — is NOT covered by LM Studio's docs (they
# only document a systemd startup task for Linux); it's this script's own
# best-effort design using Windows Task Scheduler, the closest equivalent to
# the systemd --user service confirmed working on Linux. None of this has
# actually been run on a Windows machine — this session only had a Linux
# (Bazzite) box to test against. Review before relying on it.

Write-Host "Installing llmster (LM Studio headless) via the official installer..."
irm https://lmstudio.ai/install.ps1 | iex

$lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
if (-not (Test-Path $lms)) {
    Write-Error "lms.exe not found at $lms after install — check the installer actually completed."
    exit 1
}

Write-Host "Starting the llmster daemon and API server once, to confirm it works..."
& $lms daemon up
Start-Sleep -Seconds 2
& $lms server start
Write-Host "Check http://localhost:1234/v1/models — it should list at least one model."

# ── Persistence: a logon-triggered Scheduled Task (Windows' closest
# equivalent of the systemd --user service used on Linux). Not something LM
# Studio's own docs describe — a Windows Service would be an alternative if
# this proves unreliable, but a logon task is simpler and doesn't need a
# separate service-wrapper tool (e.g. NSSM) as a dependency. ─────────────────
$wrapperPath = Join-Path $env:USERPROFILE ".lmstudio\lmstudio-start.ps1"
@"
& '$lms' daemon up
Start-Sleep -Seconds 2
& '$lms' server start
"@ | Set-Content -Path $wrapperPath -Encoding UTF8

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$wrapperPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "LLM-CODER LM Studio (llmster)" `
    -Action $action -Trigger $trigger -Settings $settings -Force

Write-Host "Scheduled Task registered — llmster will now start automatically at logon."
Write-Host "Verify manually with: Start-ScheduledTask -TaskName 'LLM-CODER LM Studio (llmster)'"
