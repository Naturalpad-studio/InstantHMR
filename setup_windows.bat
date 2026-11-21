@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

rem ============================================================
rem  InstantHMR - one-click Windows setup
rem
rem  setup_windows.bat                         create .venv, install CUDA torch
rem                                            + requirements, pre-download models
rem  setup_windows.bat --check                 report what is already installed
rem  setup_windows.bat --proxy <url>           also use an HTTP proxy for pip/HF
rem  setup_windows.bat --model medium --fit 2  pre-download another model variant
rem ============================================================

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "CHECK=0"
set "MODEL=medium-commercial"
set "FIT=2"
set "PROXY="

:parse
if "%~1"=="" goto run
if /i "%~1"=="--check" (
  set "CHECK=1"
  shift
  goto parse
)
if /i "%~1"=="--proxy" (
  set "PROXY=%~2"
  shift
  shift
  goto parse
)
if /i "%~1"=="--model" (
  set "MODEL=%~2"
  shift
  shift
  goto parse
)
if /i "%~1"=="--fit" (
  set "FIT=%~2"
  shift
  shift
  goto parse
)
echo [WARN] unknown option "%~1", ignored
shift
goto parse

:run
if defined PROXY (
  set "HTTP_PROXY=%PROXY%"
  set "HTTPS_PROXY=%PROXY%"
  set "http_proxy=%PROXY%"
  set "https_proxy=%PROXY%"
  echo [INFO] proxy: %PROXY%
)

echo ============================================================
echo  InstantHMR setup      root: %ROOT%
echo ============================================================
echo.

rem ---------- 1. interpreter ----------
if not exist "%PY%" goto no_venv
"%PY%" -c "import sys;print('[OK]   venv python', sys.version.split()[0])"
goto python_ready

:no_venv
if "%CHECK%"=="1" goto no_venv_check
echo [INFO] no .venv, creating one with py -3.12
py -3.12 -c "import sys" >nul 2>&1
if errorlevel 1 goto no_python
py -3.12 -m venv "%ROOT%.venv"
if errorlevel 1 goto venv_failed
echo [OK]   .venv created
goto python_ready

:no_python
echo [ERROR] Python 3.12 not found - install it from https://www.python.org/downloads/
echo         (any of 3.10 - 3.12 works: py -3.12 -m venv .venv)
goto fail

:venv_failed
echo [ERROR] could not create .venv
goto fail

:no_venv_check
echo [MISS] .venv not created yet
goto check_only

:python_ready
if "%CHECK%"=="1" goto check_only

rem ---------- 2. packages ----------
echo.
echo [1/3] installing torch + torchvision (CUDA 12.6)
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo [WARN] CUDA wheels failed - falling back to CPU wheels, the detector will run on CPU
  "%PY%" -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 (
    echo [ERROR] torch install failed
    goto fail
  )
)

echo.
echo [2/3] installing requirements.txt
"%PY%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
  echo [ERROR] requirements install failed
  goto fail
)

rem ---------- 3. model weights ----------
echo.
echo [3/3] pre-downloading ONNX graphs into the Hugging Face cache
"%PY%" "%ROOT%tools\prefetch_models.py" --model %MODEL% --fit %FIT%
if errorlevel 1 echo [WARN] model download failed - demo.py will retry on first run

rem ---------- 4. report ----------
:check_only
echo.
echo --------------------- environment check ---------------------
if not exist "%PY%" (
  echo [MISS] .venv       : not created
  goto endreport
)
"%PY%" -c "import torch;print('[OK]   torch       :',torch.__version__,'| cuda:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>nul
if errorlevel 1 echo [MISS] torch       : not installed
"%PY%" -c "import onnxruntime as o;print('[OK]   onnxruntime :',o.__version__,'|',[p for p in o.get_available_providers() if p not in ('CPUExecutionProvider',)])" 2>nul
if errorlevel 1 echo [MISS] onnxruntime : not installed
"%PY%" -c "import rfdetr,rerun;print('[OK]   rfdetr/rerun:',rfdetr.__version__ if hasattr(rfdetr,'__version__') else 'ok')" 2>nul
if errorlevel 1 echo [MISS] rfdetr/rerun: not installed

:endreport
echo -------------------------------------------------------------
echo  Next:  run_image.bat ^| run_video.bat clip.mp4 ^| run_camera.bat 0
echo  Docs:  WINDOWS.md  (model list, parameters, troubleshooting)
echo.
if "%CHECK%"=="1" (
  pause >nul
  endlocal
  exit /b 0
)
echo [DONE] setup finished.
pause >nul
endlocal
exit /b 0

:fail
echo.
echo [FAILED] setup stopped - fix the message above and run setup_windows.bat again.
pause >nul
endlocal
exit /b 1
