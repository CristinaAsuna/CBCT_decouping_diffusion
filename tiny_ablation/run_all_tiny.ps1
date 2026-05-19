param(
    [string]$CaseRoot = "D:\nnunet\2d_projection_physics_consistent",
    [string]$CaseList = "tiny_ablation/case_splits/tiny_cases.txt",
    [string]$OutputRoot = "tiny_ablation/outputs",
    [int]$NumCases = 16,
    [int]$MaxSteps = 5000,
    [int]$SamplerSteps = 20,
    [switch]$DryRun,
    [switch]$EnableSwanlab,
    [switch]$NoTensorboard
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$caseArgs = @(
    "tiny_ablation/scripts/make_tiny_cases.py",
    "--case-root", $CaseRoot,
    "--out", $CaseList,
    "--num-cases", "$NumCases"
)

Write-Host "Creating tiny case list..."
python @caseArgs

$runArgs = @(
    "tiny_ablation/scripts/run_tiny_ablation.py",
    "--case-root", $CaseRoot,
    "--case-names-file", $CaseList,
    "--output-root", $OutputRoot,
    "--max-steps", "$MaxSteps",
    "--sampler-steps", "$SamplerSteps"
)

if ($DryRun) {
    $runArgs += "--dry-run"
}
if ($EnableSwanlab) {
    $runArgs += "--enable-swanlab"
}
if ($NoTensorboard) {
    $runArgs += "--no-tensorboard"
}

Write-Host "Running tiny ablations..."
python @runArgs
