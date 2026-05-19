param(
    [string]$CaseRoot = "D:/nnunet/2d_projection_physics_consistent",
    [string]$CaseNamesFile = "tiny_ablation/case_splits/tiny_cases.txt",
    [string]$OutputRoot = "tiny_ablation/outputs",
    [string]$RunTag = "",
    [int]$NumCases = 16,
    [int]$MaxSteps = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = "branch_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

Write-Host "Branch tiny run tag: $RunTag"
Write-Host "Outputs will be written under: $OutputRoot/$RunTag"

python tiny_ablation/scripts/make_tiny_cases.py `
    --case-root $CaseRoot `
    --out $CaseNamesFile `
    --num-cases $NumCases

$cmd = @(
    "tiny_ablation/scripts/run_branch_tiny_ablation.py",
    "--case-root", $CaseRoot,
    "--case-names-file", $CaseNamesFile,
    "--output-root", $OutputRoot,
    "--run-tag", $RunTag
)

if ($MaxSteps -gt 0) {
    $cmd += @("--max-steps", "$MaxSteps")
}

if ($DryRun) {
    $cmd += "--dry-run"
}

python @cmd
