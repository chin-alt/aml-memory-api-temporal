param(
    [ValidateSet('smoke16', 'pilot120', 'full')][string]$Experiment = 'pilot120',
    [ValidateRange(1, 100)][int]$TopK = 100,
    [ValidateRange(1, 10)][int]$WriteConcurrency = 3,
    [ValidateRange(1, 32)][int]$QaConcurrency = 3,
    [ValidateRange(60, 14400)][int]$MemoryHttpTimeoutSeconds = 3600,
    [string]$EnvFile = '',
    [switch]$Run,
    [string]$Resume = '',
    [string]$OnlyConversations = '',
    [switch]$RecoverVectorStore,
    [switch]$SkipFailed,
    [switch]$AllowHttpModels
)
. (Join-Path $PSScriptRoot '_common.ps1')
$profile = Initialize-BenchmarkRuntime
$arguments = @('-m', 'aml_adapter.locomo_refined_batch', '--memory-profile', $profile,
    '--experiment', $Experiment, '--top-k', "$TopK", '--write-concurrency', "$WriteConcurrency",
    '--qa-concurrency', "$QaConcurrency", '--memory-http-timeout-seconds', "$MemoryHttpTimeoutSeconds",
    '--seed', '42', '--env-file', (Resolve-BenchmarkEnvFile $EnvFile))
if ($Run) { $arguments += '--run' }
if ($Resume) { $arguments += @('--resume', $Resume) }
if ($OnlyConversations) { $arguments += @('--only-conversations', $OnlyConversations) }
if ($RecoverVectorStore) { $arguments += '--recover-vector-store' }
if ($SkipFailed) { $arguments += '--skip-failed' }
if ($AllowHttpModels) { $arguments += '--allow-http-models' }
& $script:BenchmarkPython @arguments
exit $LASTEXITCODE
