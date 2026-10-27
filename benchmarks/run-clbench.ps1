param(
    [ValidateRange(1, 20)][int]$PerCategory = 1,
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
$arguments = @('-m', 'aml_adapter.clbench_batch', '--memory-profile', $profile,
    '--per-category', "$PerCategory", '--sampling', $Sampling, '--top-k', "$TopK",
    '--env-file', (Resolve-BenchmarkEnvFile $EnvFile))
if ($Run) { $arguments += '--run' }
if ($Resume) { $arguments += @('--resume', $Resume) }
if ($SkipFailed) { $arguments += '--skip-failed' }
if ($AllowHttpModels) { $arguments += '--allow-http-models' }
& $script:BenchmarkPython @arguments
exit $LASTEXITCODE
