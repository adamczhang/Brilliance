@echo off
rem Brilliance: rad-export (a finetune to a .rad with the stock model's recipe) is a Python 3 script,
rem tools/rad-export, installed beside this file. This runs it with the Python the machine has: the
rem py launcher first, else python on PATH. Folding a LoRA adapter needs torch and safetensors in that
rem Python; listing recipes and converting a merged checkpoint need only the standard library.
rem rad-export finds rad-convert on PATH. A package's bin\ (this folder) is not on PATH when the ZIP
rem is unpacked, so this folder goes first: the rad-convert beside this file is the one it runs, and
rem its recipes (..\share\radiance\recipes) are the ones it lists. --rad-convert PATH still chooses
rem another.
rem The exit status is rad-export's, so a failed export fails its caller. No parenthesised block
rem around the run: %errorlevel% inside one is read when the block is parsed, before Python runs,
rem and a bare `exit /b` there gave PowerShell 0 after a failure.
setlocal
set "PATH=%~dp0;%PATH%"
set "BRL_RAD_EXPORT_PY="
where py >nul 2>nul && set "BRL_RAD_EXPORT_PY=py -3"
if not defined BRL_RAD_EXPORT_PY where python >nul 2>nul && set "BRL_RAD_EXPORT_PY=python"
if not defined BRL_RAD_EXPORT_PY echo rad-export: no Python 3 found (neither the py launcher nor python on PATH) 1>&2
if not defined BRL_RAD_EXPORT_PY exit /b 9009
%BRL_RAD_EXPORT_PY% "%~dp0rad-export" %*
exit /b %errorlevel%
