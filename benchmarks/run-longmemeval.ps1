param(
    [ValidateRange(1, 6)][int]$Concurrency = 1,
    [string]$EnvFile = '',
    [switch]$Run,
    [string]$Resume = '',
    [switch]$SkipFailed,
    [switch]$AllowHttpModels
)
. (Join-Path $PSScriptRoot '_common.ps1')
$profile = Initialize-BenchmarkRuntime
$envFilePath = Resolve-BenchmarkEnvFile $EnvFile
& $script:BenchmarkPython (Join-Path $PSScriptRoot 'prepare_longmemeval.py')
if ($LASTEXITCODE -ne 0) { throw 'LongMemEval data preparation failed.' }
$arguments = @('-m', 'aml_adapter.lme_batch', '--experiment', 'oracle5',
    '--memory-profile', $profile, '--concurrency', "$Concurrency", '--env-file', $envFilePath)
if ($Run) { $arguments += '--run' }
if ($Resume) { $arguments += @('--resume', $Resume) }
if ($SkipFailed) { $arguments += '--skip-failed' }
if ($AllowHttpModels) { $arguments += '--allow-http-models' }
& $script:BenchmarkPython @arguments
exit $LASTEXITCODE
