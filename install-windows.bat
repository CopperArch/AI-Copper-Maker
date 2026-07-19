@echo off
REM Double-click installer — runs install-windows.ps1 with a bypassed
REM execution policy so it works even on a machine that has never allowed
REM scripts before.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows.ps1"
pause
