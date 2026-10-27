param(
    [ValidateSet('both', 'mcq', 'generative')][string]$Mode = 'both',
    [ValidateRange(1, 100)][int]$TopK = 100,
    [string]$QaCutoffs = '10,20,50,100',
    [ValidateRange(5, 1000)][int]$PerUser = 5,
    [string]$EnvFile = '',
    [switch]$Run,
    [string]$Resume = '',
    [switch]$SkipFailed,
    [switch]$AllowHttpModels
)
. (Join-Path $PSScriptRoot '_common.ps1')
$profile = Initialize-BenchmarkRuntime
$arguments = @('-m', 'aml_adapter.persona_batch', '--memory-profile', $profile, '--mode', $Mode,
    '--top-k', "$TopK", '--qa-cutoffs', $QaCutoffs, '--mcq-format', 'original',
    '--extend-per-user', "$PerUser", '--env-file', (Resolve-BenchmarkEnvFile $EnvFile))
if ($Run) { $arguments += '--run' }
if ($Resume) { $arguments += @('--resume', $Resume) }
if ($SkipFailed) { $arguments += '--skip-failed' }
if ($AllowHttpModels) { $arguments += '--allow-http-models' }
& $script:BenchmarkPython @arguments
exit $LASTEXITCODE
