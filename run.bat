@echo off
setlocal

REM --- Path to your virtual environment ---
set "VENV=C:\Users\Hellx\Documents\Programming\python\Project\dgpo_project\venv"

REM --- Detect if running in PowerShell or CMD ---
set "PSCHECK=%PSModulePath%"
if defined PSCHECK (
    echo [INFO] Detected PowerShell. Activating PowerShell venv...
    powershell -NoExit -ExecutionPolicy Bypass -Command "& '%VENV%\Scripts\Activate.ps1'"
) else (
    echo [INFO] Detected CMD. Activating CMD venv...
    call "%VENV%\Scripts\activate.bat"
)

endlocal
