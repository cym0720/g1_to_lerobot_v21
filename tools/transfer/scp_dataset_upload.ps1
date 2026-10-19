param(
    [Parameter(Mandatory = $true)][string]$SourcePath,
    [Parameter(Mandatory = $true)][string]$DatasetName,
    [Parameter(Mandatory = $true)][string]$LogPath
)

$ErrorActionPreference = 'Stop'
$sshConfig = 'C:\Users\cym666\.ssh\config'
$sshExe = 'D:\OpenSSH-Win64\OpenSSH-Win64\ssh.exe'
$scpExe = 'D:\OpenSSH-Win64\OpenSSH-Win64\scp.exe'
$remoteAlias = 'A100'
$remoteRoot = '/data/tzq/datasets/starVLA/Datasets/0923_20s_20a'
$exitCode = 1

try {
    if ($DatasetName -notmatch '^[A-Za-z0-9_-]+$') {
        throw "Invalid dataset folder name: $DatasetName"
    }
    $source = (Resolve-Path -LiteralPath $SourcePath).Path
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Dataset folder not found: $source"
    }
    if (Test-Path -LiteralPath $LogPath) {
        throw "Refusing to overwrite transfer log: $LogPath"
    }
    if (-not (Test-Path -LiteralPath $sshConfig -PathType Leaf)) {
        throw "SSH config not found: $sshConfig"
    }
    if (-not (Test-Path -LiteralPath $sshExe -PathType Leaf) -or
        -not (Test-Path -LiteralPath $scpExe -PathType Leaf)) {
        throw 'OpenSSH ssh.exe/scp.exe are unavailable at the configured path.'
    }

    $remoteDatasetPath = "$remoteRoot/$DatasetName"
    $remoteTarget = "${remoteAlias}:$remoteRoot/"
    $files = @(Get-ChildItem -LiteralPath $source -Recurse -File)
    $expectedCount = $files.Count
    $expectedBytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $expectedBytes) { $expectedBytes = 0 }

    Start-Transcript -LiteralPath $LogPath | Out-Null
    $Host.UI.RawUI.WindowTitle = "SCP upload - $DatasetName"
    Write-Host "Source: $source"
    Write-Host "Destination: $remoteTarget"
    Write-Host "Local files: $expectedCount; bytes: $expectedBytes"

    & $sshExe -F $sshConfig -o BatchMode=yes -o ConnectTimeout=15 $remoteAlias "test ! -e '$remoteDatasetPath'"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote destination already exists or cannot be checked: $remoteDatasetPath"
    }

    & $scpExe -F $sshConfig -r -p $source $remoteTarget
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed with exit code $LASTEXITCODE"
    }

    $verifyCommand = "test -d '$remoteDatasetPath' && printf 'REMOTE_FILE_COUNT=' && find '$remoteDatasetPath' -type f | wc -l && printf 'REMOTE_BYTE_COUNT=' && du -sb '$remoteDatasetPath' | cut -f1"
    $verification = @(& $sshExe -F $sshConfig -o BatchMode=yes -o ConnectTimeout=15 $remoteAlias $verifyCommand)
    if ($LASTEXITCODE -ne 0) {
        throw 'SCP returned success, but the remote verification command failed.'
    }
    $verification | ForEach-Object { Write-Host $_ }
    $remoteCount = [int](($verification | Where-Object { $_ -match '^REMOTE_FILE_COUNT=' } | Select-Object -First 1) -replace '^REMOTE_FILE_COUNT=', '').Trim()
    $remoteBytes = [long](($verification | Where-Object { $_ -match '^REMOTE_BYTE_COUNT=' } | Select-Object -First 1) -replace '^REMOTE_BYTE_COUNT=', '').Trim()
    if ($remoteCount -ne $expectedCount -or $remoteBytes -ne $expectedBytes) {
        throw "Remote verification mismatch: expected $expectedCount files/$expectedBytes bytes, got $remoteCount files/$remoteBytes bytes."
    }

    Write-Host "TRANSFER_OK $DatasetName"
    $exitCode = 0
}
catch {
    Write-Host "TRANSFER_FAILED $DatasetName : $_" -ForegroundColor Red
}
finally {
    if (Get-Command Stop-Transcript -ErrorAction SilentlyContinue) {
        try { Stop-Transcript | Out-Null } catch { }
    }
    Read-Host 'Transfer finished. Press Enter to close this window' | Out-Null
}

exit $exitCode
