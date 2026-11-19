#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: start Open WebUI headless against a local server, for openwebui_check.py.
.DESCRIPTION
  Auth off, one OpenAI connection to -Server (no key), Ollama off, offline (no version check, no
  pip installs for tools), and an OpenAI embedding engine so no embedding model is downloaded. Open
  WebUI copies these settings into its database on the first start and ignores them after, so each
  run gets a fresh -DataDir unless one is given. Prints the process id; stop it with Stop-Process
  (it is a client, not the server).
.EXAMPLE
  powershell -File win\tools\openwebui-start.ps1 -Venv out\openwebui-venv
#>
param(
    [Parameter(Mandatory)][string]$Venv,
    [string]$Server = 'http://127.0.0.1:8100/v1',
    [int]$Port = 8180,
    [string]$DataDir = '',
    [string]$Log = ''
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path (Resolve-Path $Venv) 'Scripts\open-webui.exe'
if (-not (Test-Path $exe)) { throw "no open-webui.exe under $Venv" }
if (-not $DataDir) { $DataDir = Join-Path (Split-Path (Resolve-Path $Venv)) ("openwebui-data-" + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $DataDir | Out-Null
if (-not $Log) { $Log = Join-Path $DataDir 'openwebui.log' }
$env:DATA_DIR = $DataDir
$env:WEBUI_AUTH = 'False'
$env:WEBUI_SECRET_KEY = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
$env:ENABLE_OLLAMA_API = 'False'
$env:OPENAI_API_BASE_URL = $Server
$env:OPENAI_API_KEY = 'none'
$env:RAG_EMBEDDING_ENGINE = 'openai'
$env:RAG_OPENAI_API_BASE_URL = $Server
$env:OFFLINE_MODE = 'true'
$env:PYTHONIOENCODING = 'utf-8'
$p = Start-Process -FilePath $exe -ArgumentList 'serve', '--host', '127.0.0.1', '--port', "$Port" `
    -WorkingDirectory $DataDir -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
    -WindowStyle Hidden -PassThru
$up = $false
foreach ($i in 1..180) {
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health" -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch {}
    if ($p.HasExited) { break }
    Start-Sleep -Seconds 1
}
if (-not $up) { Write-Host "open-webui did not come up (exited: $($p.HasExited)); see $Log.err"; exit 1 }
Write-Host "open-webui $($p.Id) on http://127.0.0.1:$Port, data $DataDir"
$p.Id
