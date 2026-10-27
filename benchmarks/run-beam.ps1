param(
    [ValidateRange(1, 20)][int]$Conversations = 1,
    [ValidateRange(1, 2)][int]$PerType = 1,
    [ValidateSet('smoke', 'stratified')][string]$Sampling = 'smoke',
    [ValidateRange(1, 100)][int]$TopK = 100,
    [string]$EnvFile = '',
    [switch]$Run,
    [string]$Resume = '',
    [switch]$SkipFailed,
    [switch]$AllowHttpModels
)
. (Join-Path $PSScriptRoot '_common.ps1')
$profile = Initialize-BenchmarkRuntime
if ($Run) {
    $compose = Join-Path $script:BenchmarkAdapterRoot 'deploy/beam/docker-compose.yml'
    & docker compose -f $compose up -d
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start the isolated BEAM Milvus stack.' }
    $healthy = $false
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:9191/healthz' -TimeoutSec 3
            if ($response.StatusCode -eq 200) { $healthy = $true; break }
        } catch { }
        Start-Sleep -Seconds 3
    }
    if (-not $healthy) { throw 'Isolated BEAM Milvus did not become healthy within 180 seconds.' }
}
$arguments = @('-m', 'aml_adapter.beam_batch', '--memory-profile', $profile, '--size', '100K',
    '--conversations', "$Conversations", '--per-type', "$PerType", '--sampling', $Sampling,
    '--top-k', "$TopK", '--seed', '42', '--env-file', (Resolve-BenchmarkEnvFile $EnvFile))
if ($Run) { $arguments += '--run' }
if ($Resume) { $arguments += @('--resume', $Resume) }
if ($SkipFailed) { $arguments += '--skip-failed' }
if ($AllowHttpModels) { $arguments += '--allow-http-models' }
& $script:BenchmarkPython @arguments
exit $LASTEXITCODE
