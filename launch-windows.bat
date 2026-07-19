@echo off
REM Double-click launcher — runs launch-windows.ps1 with a bypassed execution
REM policy so it works even on a machine that has never allowed scripts
REM before (the installer's Start Menu/Desktop shortcuts invoke this).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-windows.ps1"
