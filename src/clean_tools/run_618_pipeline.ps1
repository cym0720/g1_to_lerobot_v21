param(
    [string]$InputRoot = "dataset/raw/g1/618data",
    [string]$Contract = "config/contracts/eva_replay_g1_qpos20_v1.json",
    [ValidateSet("wrist_lr")]
    [string]$CameraStandard = "wrist_lr",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$DatasetName = "618data",
    [string]$ConvertedRoot,
    [string]$OutputRoot = "output/current",
    [ValidateSet("raw_encoder_0_120", "opening_0_1", "closed_0_1")]
    [string]$GripperConvention,
    [switch]$AllowStateActionFallback,
    [switch]$EnableTrim,
    [switch]$EnableActionSmoothing,
    [int]$Workers = 8,
    [double]$MinDurationSeconds = 0.0,
    [double]$MotionThreshold = 0.001
)

# 普通 G1 主路径：转换基线和 output/current 中间副本均按数据标准/相机标准分层。
# 通过清洗契约的最终数据另写 output/cleaned；乒乓球仅在公共校验尾部前插入检测和裁剪。
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "缺少 uv 管理的环境：$python。请先在仓库根目录执行：uv sync --extra dev"
}
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $Contract)) { throw "Contract not found: $Contract" }
$dataStandard = (Get-Content -LiteralPath $Contract -Raw | ConvertFrom-Json).contract_id
if ([string]::IsNullOrWhiteSpace($dataStandard) -or $dataStandard -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "Contract must define a safe contract_id: $Contract"
}
$standardCameraRoot = Join-Path $repoRoot (Join-Path "dataset/converted" (Join-Path $dataStandard $CameraStandard))
if ([string]::IsNullOrWhiteSpace($ConvertedRoot)) {
    $ConvertedRoot = Join-Path $standardCameraRoot $DatasetName
} else {
    $convertedRootCandidate = $ConvertedRoot
    if (-not [System.IO.Path]::IsPathRooted($convertedRootCandidate)) {
        $convertedRootCandidate = Join-Path $repoRoot $convertedRootCandidate
    }
    $resolvedConvertedRoot = [System.IO.Path]::GetFullPath($convertedRootCandidate)
    $resolvedStandardCameraRoot = [System.IO.Path]::GetFullPath($standardCameraRoot).TrimEnd('\') + '\'
    if (-not $resolvedConvertedRoot.StartsWith($resolvedStandardCameraRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "ConvertedRoot must be under dataset/converted/<contract_id>/<camera-standard>/: $ConvertedRoot"
    }
    $ConvertedRoot = $resolvedConvertedRoot
}
$runOutputRoot = Join-Path $OutputRoot (Join-Path $dataStandard $CameraStandard)
$dataRoot = Join-Path $runOutputRoot "data"
$reportsRoot = Join-Path $runOutputRoot "reports"

if (Test-Path $ConvertedRoot) { throw "ConvertedRoot already exists: $ConvertedRoot" }
if (Test-Path $dataRoot) { throw "Output data already exists: $dataRoot" }
New-Item -ItemType Directory -Path $runOutputRoot -Force | Out-Null

$convertArgs = @(
    "tools/conversion/00_convert_618_to_lerobot.py", "--input", $InputRoot,
    "--output", $ConvertedRoot, "--contract", $Contract,
    "--workers", $Workers, "--min-duration-s", $MinDurationSeconds
)
if ($GripperConvention) { $convertArgs += @("--gripper-convention", $GripperConvention) }
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
& $python tools/testing/01_validate_lerobot_v21_read.py --input $dataRoot --workers $Workers --report (Join-Path $reportsRoot "read_validation.csv")
$parquet = Get-ChildItem -Path $dataRoot -Recurse -Filter "episode_*.parquet" | Select-Object -First 1
if ($null -eq $parquet) { throw "No converted episode written" }
& $python tools/testing/act_direct_loss_smoke.py --parquet $parquet.FullName --output (Join-Path $reportsRoot "act_loss_smoke.txt")
