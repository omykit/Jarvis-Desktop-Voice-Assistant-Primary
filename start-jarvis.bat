@echo off
cd /d "%~dp0"
set "LOGFILE=%~dp0jarvis_startup.log"
echo [%date% %time%] Jarvis startup requested>>"%LOGFILE%"

if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0jarvis_desktop.py"
    echo [%date% %time%] Launched with project .venv pythonw.exe>>"%LOGFILE%"
    exit /b 0
)

if exist "C:\Python313\pythonw.exe" (
    start "" "C:\Python313\pythonw.exe" "%~dp0jarvis_desktop.py"
    echo [%date% %time%] Launched with C:\Python313\pythonw.exe>>"%LOGFILE%"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python313\pythonw.exe" (
    start "" "%LocalAppData%\Programs\Python\Python313\pythonw.exe" "%~dp0jarvis_desktop.py"
    echo [%date% %time%] Launched with LocalAppData pythonw.exe>>"%LOGFILE%"
    exit /b 0
)

where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw "%~dp0jarvis_desktop.py"
    echo [%date% %time%] Launched with PATH pythonw>>"%LOGFILE%"
    exit /b 0
)

where py >nul 2>nul
if not errorlevel 1 (
    start "Jarvis Launcher" cmd /c py "%~dp0jarvis_desktop.py" >>"%LOGFILE%" 2>&1
    echo [%date% %time%] Fell back to py launcher>>"%LOGFILE%"
    exit /b 0
)

echo [%date% %time%] No Python launcher found>>"%LOGFILE%"
