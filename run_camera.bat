@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

rem ============================================================
rem  InstantHMR - live camera visualization (Rerun viewer)
rem  Usage:  run_camera.bat [camera_index] [extra demo.py args]
rem  Example: run_camera.bat            (default camera 0)
rem           run_camera.bat 1
rem           run_camera.bat 0 --detector-stride 3 --max-persons 2
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

if not defined HMR_MODEL set "HMR_MODEL=%ROOT%models\instanthmr-medium-commercial.onnx"
if not defined HMR_DEVICE set "HMR_DEVICE=cuda"
if not defined HMR_DETECTOR set "HMR_DETECTOR=nano"

if "%~1"=="" (
  set "ARGS=0"
  echo [INFO] no camera index given, using camera 0
) else (
  set "ARGS=%*"
)

if not exist "%HMR_MODEL%" (
  echo [ERROR] model not found: %HMR_MODEL%
  pause
  exit /b 1
)

cd /d "%ROOT%"
echo [INFO] model   : %HMR_MODEL%
echo [INFO] device  : %HMR_DEVICE%   detector: %HMR_DETECTOR%
echo [INFO] pose filter: one-euro (override with --pose-filter alpha-beta / --smooth)
echo [INFO] Ctrl+C in this window stops the demo.
echo.

rem --detector-stride 3: detector every 3rd frame, boxes follow the 2D keypoints in between
rem (README's recommendation for live demos; override it by appending your own --detector-stride)
"%PY%" demo.py --fit 0 --model "%HMR_MODEL%" --device %HMR_DEVICE% --detector %HMR_DETECTOR% --detector-stride 3 --pose-filter one-euro --camera !ARGS!

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] demo exited with code %RC%
echo [INFO] finished. Close this window or press any key.
pause >nul
endlocal
