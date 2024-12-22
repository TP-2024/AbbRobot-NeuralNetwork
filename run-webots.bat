@echo off

IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) ELSE (
    echo Virtual environment not found
    exit /b 1
)

set "WEBOTS_HOME=C:\Program Files\Webots"
set "QT_PLUGIN_PATH=C:\Program Files\Webots\msys64\mingw64\plugins"

start "" "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
