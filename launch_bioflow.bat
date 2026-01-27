@echo off
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║   🧬 BioFlow - AI-Powered Drug Discovery Platform        ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo Starting BioFlow UI (Next.js)...
echo.

cd /d "%~dp0"
if not exist "ui\package.json" (
  echo ❌ Error: Next.js UI not found at .\ui
  echo Run `launch_bioflow_full.bat` from the repo root.
  pause
  exit /b 1
)

cd /d "%~dp0\ui"
pnpm dev

pause
