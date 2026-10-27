$ErrorActionPreference = 'Stop'
$script:BenchmarkRepoRoot = Split-Path $PSScriptRoot -Parent
$script:BenchmarkAdapterRoot = Join-Path $script:BenchmarkRepoRoot 'adapter'
$script:BenchmarkMemoryRoot = Join-Path $script:BenchmarkRepoRoot 'memory'
$script:BenchmarkPython = Join-Path $script:BenchmarkRepoRoot '.venv/Scripts/python.exe'

function Initialize-BenchmarkRuntime {
    if (-not (Test-Path -LiteralPath $script:BenchmarkPython)) {
        throw 'Benchmark virtual environment is missing. Run benchmarks/setup.ps1 first.'
    }
    $profilePath = Join-Path $PSScriptRoot 'profile.txt'
    if (-not (Test-Path -LiteralPath $profilePath)) { throw 'benchmarks/profile.txt is missing.' }
    $profile = (Get-Content -LiteralPath $profilePath -Raw).Trim()
    if ($profile -notin @('ordinary', 'schema-entity', 'schema-temporal')) {
        throw "Unsupported benchmark profile: $profile"
    }
    $env:PYTHONPATH = "$script:BenchmarkMemoryRoot;$script:BenchmarkAdapterRoot"
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONHASHSEED = '42'
    return $profile
}

function Resolve-BenchmarkEnvFile([string]$Value) {
    if ($Value) { return [System.IO.Path]::GetFullPath($Value) }
    return Join-Path $script:BenchmarkAdapterRoot 'config/longmemeval.local.env'
}
