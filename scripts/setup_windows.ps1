$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDirectory
Set-Location $ProjectRoot

$PythonCommand = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = @("py", "-3.11")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = @("python")
} else {
    throw "Khong tim thay Python 3.11. Hay cai Python 3.11 x64 truoc."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv .venv
    } else {
        & $PythonCommand[0] -m venv .venv
    }
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e ".[dev]"
& $VenvPython -m pytest -q

New-Item -ItemType Directory -Force -Path "artifacts\environment" | Out-Null
& $VenvPython -m dss_stage6.environment --output "artifacts\environment\local_environment.json"

Write-Host "Setup hoan tat. Tao configs\paths.local.yaml truoc khi chay preflight."

