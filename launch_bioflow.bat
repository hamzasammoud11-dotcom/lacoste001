@echo off
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║   🧬 BioFlow - AI-Powered Drug Discovery Platform        ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo Starting BioFlow UI...
echo.

cd /d "%~dp0"
python -m streamlit run bioflow/ui/app.py --server.port 8501

pause
