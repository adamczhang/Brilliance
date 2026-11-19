@echo off
rem Brilliance: rad-export (a finetune to a .rad with the stock model's recipe) is a Python 3 script,
rem tools/rad-export, installed beside this file. This runs it with the Python the machine has: the
rem py launcher first, else python on PATH. Folding a LoRA adapter needs torch and safetensors in that
rem Python; listing recipes and converting a merged checkpoint need only the standard library.
setlocal
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0rad-export" %*
  exit /b
)
where python >nul 2>nul
if not errorlevel 1 (
  python "%~dp0rad-export" %*
  exit /b
)
echo rad-export: no Python 3 found (neither the py launcher nor python on PATH) 1>&2
exit /b 9009
