# One-time Python environment setup on a new machine (about 3 GB download).
# Recreates .analysis-tools/python (core) and .analysis-tools/extra (TabPFN etc.)
# with the exact versions used on the home laptop.  Needs Python 3.12 (64-bit).
# Usage:
#   powershell -ExecutionPolicy Bypass -File <repo>\tools\setup_env.ps1 [-Python "C:\path\python.exe"] [-Gpu]
param([string]$Python = "python", [switch]$Gpu)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root ".analysis-tools"
& $Python --version

$env:PYTHONPATH = ""
& $Python -m pip install --upgrade --target (Join-Path $Tools "python") -r (Join-Path $PSScriptRoot "requirements_core.txt")
& $Python -m pip install --no-deps --target (Join-Path $Tools "msvc") msvc-runtime
& $Python -m pip install --upgrade --target (Join-Path $Tools "extra") --extra-index-url https://download.pytorch.org/whl/cpu -r (Join-Path $PSScriptRoot "requirements_extra.txt")
if ($Gpu) {
    & $Python -m pip install --no-deps --target (Join-Path $Tools "extra_gpu") torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
}
Write-Host "Environment ready.  Run scripts as:  cd research; `$env:PYTHONPATH=''; $Python -u <script>.py" -ForegroundColor Green
Write-Host "TabPFN v2 weights download automatically on first use (Hugging Face, Prior-Labs/TabPFN-v2-reg)." -ForegroundColor Green
