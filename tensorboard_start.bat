@echo off
REM Set the directory where SB3 writes logs
set LOGDIR="./logs/"
if "%LOGDIR%"=="" (
    echo Usage: run_tensorboard.bat path\to\logdir [port]
    pause
    goto :eof
)
REM Optional second argument: port number
set PORT=%~2
if "%PORT%"=="" set PORT=6006

echo Starting TensorBoard...
tensorboard --logdir="%LOGDIR%" --port=%PORT%
REM Optional: open browser automatically
start http://localhost:%PORT%
pause