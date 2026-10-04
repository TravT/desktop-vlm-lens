@echo off
REM ==============================================================================
REM Desktop VLM Lens - Windows 1-Click Launcher
REM ==============================================================================
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check Python virtualenv
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [INFO] No local venv detected. Running with system python...
)

REM Start MCP Server over stdio
python src\server.py
