; LLM Coder — Uncensored Edition — Windows installer.
;
; Built with NSIS (makensis). Not a self-contained app bundle: it copies the
; app's source (Python backend + HTML/JS frontend) into a per-user install
; directory and creates shortcuts. First real launch still needs
; install-windows.ps1 to be run once (installs Ollama, sets up the Python
; venv, pulls a model) — this installer offers to run it automatically at
; the end, same as any normal Windows installer's "Launch now" checkbox.
;
; Expects two command-line defines from the build script (build.sh):
;   /DSTAGE_DIR="<path to staged app source>"   (built via `git ls-files` so
;                                                 no personal data/secrets
;                                                 ever end up in the package)
;   /DAPP_VERSION="1.1.0"
;
; UNVERIFIED: this .nsi was written and compiled on Linux (via makensis) —
; there is no Windows machine in this environment to actually run the
; resulting .exe against. Review before distributing it.

!ifndef STAGE_DIR
  !error "STAGE_DIR must be defined, e.g. makensis /DSTAGE_DIR=path /DAPP_VERSION=1.1.0 installer.nsi"
!endif
!ifndef APP_VERSION
  !define APP_VERSION "1.1.0"
!endif

!include "MUI2.nsh"

Name "LLM Coder — Uncensored Edition"
OutFile "dist\LLM-Coder-Setup.exe"
Unicode true

; Per-user install under %LOCALAPPDATA% — no admin rights needed to install
; or run. The app writes its own data (conversations, config, account
; credentials) into the same directory as the code, so it must be writable
; by the current user without elevation, the same way the Linux/macOS
; install (a plain user-owned checkout) works.
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\LLM-Coder"
InstallDirRegKey HKCU "Software\LLM-Coder" "InstallDir"

VIProductVersion "${APP_VERSION}.0"
VIAddVersionKey "ProductName" "LLM Coder — Uncensored Edition"
VIAddVersionKey "FileVersion" "${APP_VERSION}.0"
VIAddVersionKey "ProductVersion" "${APP_VERSION}.0"
VIAddVersionKey "LegalCopyright" "Browns Entertainment"
VIAddVersionKey "FileDescription" "LLM Coder installer"

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\install-windows.bat"
!define MUI_FINISHPAGE_RUN_TEXT "Run first-time setup now (installs Ollama, downloads a model, sets up Python — opens a console window)"
!define MUI_FINISHPAGE_RUN_NOTCHECKED

!insertmacro MUI_PAGE_LICENSE "${STAGE_DIR}\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "LLM Coder (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /r "${STAGE_DIR}\*.*"

  WriteRegStr HKCU "Software\LLM-Coder" "InstallDir" "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\LLM Coder"
  CreateShortCut "$SMPROGRAMS\LLM Coder\LLM Coder.lnk" "$INSTDIR\launch-windows.bat" "" "$INSTDIR\launch-windows.bat" 0
  CreateShortCut "$SMPROGRAMS\LLM Coder\First-Time Setup.lnk" "$INSTDIR\install-windows.bat" "" "$INSTDIR\install-windows.bat" 0
  CreateShortCut "$SMPROGRAMS\LLM Coder\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortCut "$DESKTOP\LLM Coder.lnk" "$INSTDIR\launch-windows.bat" "" "$INSTDIR\launch-windows.bat" 0
SectionEnd

Section "Uninstall"
  MessageBox MB_YESNO|MB_ICONQUESTION \
    "This will delete the LLM Coder program folder, which also holds your saved conversations, email/calendar accounts, skills, and settings (they live alongside the code, the same way the Linux/macOS install works).$\n$\nMake sure LLM Coder isn't currently running (close its console window / the browser tab won't stop the backend), then continue to delete everything?" \
    IDYES +2
  Abort

  RMDir /r "$INSTDIR"
  Delete "$SMPROGRAMS\LLM Coder\LLM Coder.lnk"
  Delete "$SMPROGRAMS\LLM Coder\First-Time Setup.lnk"
  Delete "$SMPROGRAMS\LLM Coder\Uninstall.lnk"
  RMDir "$SMPROGRAMS\LLM Coder"
  Delete "$DESKTOP\LLM Coder.lnk"
  DeleteRegKey HKCU "Software\LLM-Coder"
SectionEnd
