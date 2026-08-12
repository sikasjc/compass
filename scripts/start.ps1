param(
    [string]$PythonCommand = "",
    [string]$UvCommand = "uv",
    [string]$EnvironmentFile = "",
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepositoryRoot

function Stop-Launch {
    param(
        [string]$Message,
        [int]$Code
    )

    [Console]::Error.WriteLine($Message)
    exit $Code
}

function Get-Utf8Text {
    param([string]$Base64)

    return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Base64))
}

if ($Port -lt 1 -or $Port -gt 65535) {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya56uv5Y+j5b+F6aG75ZyoIDEg5YiwIDY1NTM1IOS5i+mXtOOAgg==") 17
}

if (-not [string]::IsNullOrWhiteSpace($PythonCommand)) {
    try {
        Get-Command -Name $PythonCommand -ErrorAction Stop | Out-Null
        $PythonIdentity = @(
            & $PythonCommand -c "import platform,sys; print(platform.python_implementation()); print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        )
        $PythonExit = $LASTEXITCODE
    }
    catch {
        Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya6ZyA6KaBIENQeXRob24gMy4xMu+8jOivt+WFiOWuieijheW5tuWKoOWFpSBQQVRI44CC") 12
    }

    if (
        $PythonExit -ne 0 -or
        $PythonIdentity.Count -ne 2 -or
        $PythonIdentity[0].Trim() -ne "CPython" -or
        $PythonIdentity[1].Trim() -ne "3.12"
    ) {
        Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya6ZyA6KaBIENQeXRob24gMy4xMu+8jOW9k+WJjeino+mHiuWZqOS4jeespuWQiOimgeaxguOAgg==") 12
    }
}

try {
    Get-Command -Name $UvCommand -ErrorAction Stop | Out-Null
    & $UvCommand --version *> $null
    $UvExit = $LASTEXITCODE
}
catch {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya5pyq5om+5YiwIHV277yM6K+35YWI5a6J6KOFIHV2IOW5tuWKoOWFpSBQQVRI44CC") 13
}

if ($UvExit -ne 0) {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77yadXYg5peg5rOV6L+Q6KGM77yM6K+35qOA5p+l5pys5Zyw5a6J6KOF44CC") 13
}

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    # uv writes successful progress messages to stderr. Windows PowerShell 5.1
    # converts redirected native stderr into error records, so trust the process
    # exit code instead of treating that stream as a terminating exception.
    $ErrorActionPreference = "Continue"
    & $UvCommand sync *> $null
    $SyncExit = $LASTEXITCODE
}
catch {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya5L6d6LWW5ZCM5q2l5aSx6LSl44CC") 14
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}

if ($SyncExit -ne 0) {
    [Console]::Error.WriteLine(
        (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya5L6d6LWW5ZCM5q2l5aSx6LSl44CC")
    )
    exit $SyncExit
}

try {
    $PythonIdentity = @(
        & $UvCommand run python -c "import platform,sys; print(platform.python_implementation()); print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    )
    $PythonExit = $LASTEXITCODE
}
catch {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya6ZyA6KaBIENQeXRob24gMy4xMu+8jOW9k+WJjeino+mHiuWZqOS4jeespuWQiOimgeaxguOAgg==") 12
}

if (
    $PythonExit -ne 0 -or
    $PythonIdentity.Count -ne 2 -or
    $PythonIdentity[0].Trim() -ne "CPython" -or
    $PythonIdentity[1].Trim() -ne "3.12"
) {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya6ZyA6KaBIENQeXRob24gMy4xMu+8jOW9k+WJjeino+mHiuWZqOS4jeespuWQiOimgeaxguOAgg==") 12
}

$RunArguments = @("run")
if ([string]::IsNullOrWhiteSpace($EnvironmentFile)) {
    $SelectedEnvironmentFile = Join-Path $RepositoryRoot ".env"
}
else {
    $SelectedEnvironmentFile = $EnvironmentFile
}
if (Test-Path -LiteralPath $SelectedEnvironmentFile -PathType Leaf) {
    $RunArguments += "--env-file"
    $RunArguments += (Resolve-Path -LiteralPath $SelectedEnvironmentFile).Path
}
elseif (-not [string]::IsNullOrWhiteSpace($EnvironmentFile)) {
    Stop-Launch (Get-Utf8Text "5ZCv5Yqo5aSx6LSl77ya5oyH5a6a55qEIC5lbnYg5paH5Lu25LiN5a2Y5Zyo44CC") 16
}

try {
    & $UvCommand @RunArguments python -m compass.ui.app --port $Port
    $ApplicationExit = $LASTEXITCODE
}
catch [System.Management.Automation.PipelineStoppedException] {
    exit 130
}
catch {
    Stop-Launch (Get-Utf8Text "5bqU55So5ZCv5Yqo5aSx6LSl77yM6K+35p+l55yL5pys5Zyw6ISx5pWP5pel5b+X44CC") 15
}

exit $ApplicationExit
