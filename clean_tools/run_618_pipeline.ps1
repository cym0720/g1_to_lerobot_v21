param(
    [string]$InputRoot = "dataset/618data",
    [string]$OutputRoot = "output/618_cleaning_trial",
    [int]$Workers = 8,
    [double]$MinDurationSeconds = 8.0,
    [double]$MotionThreshold = 0.001
)

# 用途：先转换为原始 LeRobot v2.1，再按 LeRobot 的 Parquet/视频/meta 完整读取验证；验证通过后生成最终清洗数据集。

$ErrorActionPreference = "Stop"
$python = "python"
$armIndices = "0,1,2,3,4,5,6,7,8,9,10,11,12,13"

& $python clean_tools/00_convert_618_to_lerobot.py --input $InputRoot --output "$OutputRoot/01_lerobot_raw" --workers $Workers --min-duration-s $MinDurationSeconds
& $python clean_tools/01_validate_lerobot_v21_read.py --input "$OutputRoot/01_lerobot_raw" --workers $Workers --report "$OutputRoot/reports/02_lerobot_raw_full_read.csv"
& $python clean_tools/02_trim_arm_static_frames.py --input "$OutputRoot/01_lerobot_raw" --output "$OutputRoot/02_lerobot_cleaned" --arm-joint-indices $armIndices --threshold $MotionThreshold
& $python clean_tools/03_smooth_startup_actions.py --input "$OutputRoot/02_lerobot_cleaned"
& $python clean_tools/05_recompute_metadata_stats.py --input "$OutputRoot/02_lerobot_cleaned" --workers $Workers
# 04_review_outliers.py 按需手动运行，审查视频应放在 OutputRoot/reports/review，避免形成第四份数据集。
