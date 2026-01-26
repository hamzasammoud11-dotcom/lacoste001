@echo off
REM ============================================
REM BioFlow Full Stack Launch Script
REM ============================================
REM Launches both the FastAPI backend and Next.js frontend

echo.
echo ============================================
echo        BioFlow - AI Drug Discovery
echo ============================================
echo.

REM Check if we're in the right directory
if not exist "bioflow\api\server.py" (
    echo ERROR: Please run this script from the OpenBioMed root directory
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on port 8000...
start "BioFlow API" cmd /k "cd /d %~dp0 && python -m uvicorn bioflow.api.server:app --reload --host 0.0.0.0 --port 8000"

echo [2/2] Starting Next.js Frontend on port 3000...
timeout /t 3 /nobreak > nul
start "BioFlow UI" cmd /k "cd /d %~dp0\lacoste001\ui && pnpm dev"

echo.
echo ============================================
echo BioFlow is starting up!
echo.
echo   API:  http://localhost:8000
echo   UI:   http://localhost:3000
echo.
echo Press any key to close this window...
echo ============================================
pause > nul
