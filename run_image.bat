@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

rem ============================================================
rem  InstantHMR - image visualization (Rerun viewer)
rem  Usage:  run_image.bat [image_path] [extra demo.py args]
rem  Example: run_image.bat assets\test_zidane.jpg
rem           run_image.bat D:\photos\a.jpg --detector medium --no-mesh
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
  set "ARGS=%ROOT%assets\test_zidane.jpg"
  echo [INFO] no image given, using assets\test_zidane.jpg
) else (
  set "ARGS=%*"
)

if not exist "%HMR_MODEL%" (
  echo [ERROR] model not found: %HMR_MODEL%
  echo         export it first: tools\export_onnx.py --ckpt pytorch\medium-commercial\instanthmr-medium-commercial.pth --output models\instanthmr-medium-commercial.onnx
  pause
  exit /b 1
)

cd /d "%ROOT%"
echo [INFO] model   : %HMR_MODEL%
echo [INFO] device  : %HMR_DEVICE%   detector: %HMR_DETECTOR%
echo [INFO] opening Rerun viewer...
echo.

"%PY%" demo.py --model "%HMR_MODEL%" --device %HMR_DEVICE% --detector %HMR_DETECTOR% --fit 0 --image !ARGS!

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] demo exited with code %RC%
echo [INFO] finished. Close this window or press any key.
pause >nul
endlocal
