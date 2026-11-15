$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDirectory
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Chua co .venv. Hay chay scripts\setup_windows.ps1 truoc."
}

& $VenvPython -m pytest -q
New-Item -ItemType Directory -Force -Path "artifacts\environment" | Out-Null
& $VenvPython -m dss_stage6.environment --output "artifacts\environment\local_environment.json"
Write-Host "Environment check PASS."

