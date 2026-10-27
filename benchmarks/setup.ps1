$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & py -3.11 -m venv (Join-Path $repoRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python 3.11 virtual environment.' }
}
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
& $python -m pip install -e ((Join-Path $repoRoot 'memory') + '[deploy]') -e (Join-Path $repoRoot 'adapter')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install benchmark dependencies.' }
$example = Join-Path $repoRoot 'adapter/config/longmemeval.local.env.example'
$target = Join-Path $repoRoot 'adapter/config/longmemeval.local.env'
if (-not (Test-Path -LiteralPath $target)) {
    Copy-Item -LiteralPath $example -Destination $target
    Write-Host "Created $target; fill in the model endpoints and API keys before running."
}
Write-Host 'PASS: benchmark environment is ready.'
