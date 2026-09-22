param(
    [string]$InputRoot = "dataset/618data",
    [string]$ConvertedRoot = "dataset/converted/618_qpos20",
    [string]$OutputRoot = "output/current",
    [Parameter(Mandatory = $true)]
    [ValidateSet("raw_encoder_0_120", "opening_0_1", "closed_0_1")]
    [string]$GripperConvention,
    [switch]$AllowStateActionFallback,
    [switch]$EnableTrim,
    [switch]$EnableActionSmoothing,
    [int]$Workers = 8,
    [double]$MinDurationSeconds = 0.0,
    [double]$MotionThreshold = 0.001
)

# Standard G1 path. The immutable conversion baseline is under dataset/converted;
# output/current/data is the only maintained pipeline dataset. Pingpong inserts its
# two detector steps plus tools/split_by_ranges.py before this common validation tail.
$ErrorActionPreference = "Stop"
$python = "python"
$dataRoot = Join-Path $OutputRoot "data"

if (Test-Path $ConvertedRoot) { throw "ConvertedRoot already exists: $ConvertedRoot" }
if (Test-Path $dataRoot) { throw "Output data already exists: $dataRoot" }

$convertArgs = @(
    "tools/conversion/00_convert_618_to_lerobot.py", "--input", $InputRoot,
    "--output", $ConvertedRoot, "--workers", $Workers, "--min-duration-s", $MinDurationSeconds,
    "--gripper-convention", $GripperConvention
)
if ($AllowStateActionFallback) { $convertArgs += "--allow-state-action-fallback" }
& $python @convertArgs

# Trim and smoothing are explicit quality choices, not unconditional transformations.
if ($EnableTrim) {
    & $python src/clean_tools/02_trim_arm_static_frames.py --input $ConvertedRoot --output $dataRoot --threshold $MotionThreshold --workers $Workers
} else {
    Copy-Item -LiteralPath $ConvertedRoot -Destination $dataRoot -Recurse
}
if ($EnableActionSmoothing) {
    & $python src/clean_tools/03_smooth_startup_actions.py --input $dataRoot --workers $Workers
}

# Every material output must pass statistics, complete read/video validation, and ACT loss smoke.
& $python src/clean_tools/05_recompute_metadata_stats.py --input $dataRoot --workers $Workers
& $python tools/testing/01_validate_lerobot_v21_read.py --input $dataRoot --workers $Workers --report "$OutputRoot/reports/read_validation.csv"
$parquet = Get-ChildItem -Path $dataRoot -Recurse -Filter "episode_*.parquet" | Select-Object -First 1
if ($null -eq $parquet) { throw "No converted episode written" }
& $python tools/testing/act_direct_loss_smoke.py --parquet $parquet.FullName --output "$OutputRoot/reports/act_loss_smoke.txt"
