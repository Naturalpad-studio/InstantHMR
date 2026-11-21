@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

rem ============================================================
rem  InstantHMR - local video visualization (Rerun viewer)
rem  Usage:  run_video.bat <video_path> [extra demo.py args]
rem  Example: run_video.bat D:\videos\clip.mp4
rem           run_video.bat clip.mp4 --detector medium --detector-stride 3
rem ============================================================

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] virtual env not found: %PY%
  pause
  exit /b 1
)

rem cuDNN sub-libraries are loaded dynamically: they must be on PATH.
set "PATH=%ROOT%.venv\Scripts;%ROOT%.venv\Lib\site-packages\nvidia\cudnn\bin;%ROOT%.venv\Lib\site-packages\nvidia\cublas\bin;%ROOT%.venv\Lib\site-packages\nvidia\cuda_runtime\bin;%ROOT%.venv\Lib\site-packages\nvidia\cufft\bin;%ROOT%.venv\Lib\site-packages\nvidia\curand\bin;%ROOT%.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin;%ROOT%.venv\Lib\site-packages\nvidia\nvjitlink\bin;%PATH%"

if not defined HMR_MODEL (
  if exist "%ROOT%models\instanthmr-medium-commercial-fit2.onnx" (
    set "HMR_MODEL=%ROOT%models\instanthmr-medium-commercial-fit2.onnx"
  ) else (
    set "HMR_MODEL=%ROOT%models\instanthmr-medium-commercial.onnx"
  )
)
if not defined HMR_DEVICE set "HMR_DEVICE=cuda"
if not defined HMR_DETECTOR set "HMR_DETECTOR=nano"

if "%~1"=="" (
  echo [ERROR] no video given.
  echo         Usage: run_video.bat ^<video_path^> [extra args]
  pause
  exit /b 1
)
set "ARGS=%*"

if not exist "%HMR_MODEL%" (
  echo [ERROR] model not found: %HMR_MODEL%
  pause
  exit /b 1
)

cd /d "%ROOT%"
echo [INFO] model   : %HMR_MODEL%
echo [INFO] device  : %HMR_DEVICE%   detector: %HMR_DETECTOR%
echo [INFO] pose filter: one-euro + --smooth (override with --pose-filter alpha-beta / --no-... )
echo.

"%PY%" demo.py --model "%HMR_MODEL%" --device %HMR_DEVICE% --detector %HMR_DETECTOR% --pose-filter one-euro --smooth --video !ARGS!

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] demo exited with code %RC%
echo [INFO] finished. Close this window or press any key.
pause >nul
endlocal
