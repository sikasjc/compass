param(
    [ValidateSet("CPU", "CUDA")]
    [string]$Mode = "CUDA",
    [string]$Proxy = "",
    [switch]$IncludeDev,
    [string]$UvCommand = "uv",
    [string]$NvidiaSmiCommand = "nvidia-smi"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepositoryRoot

function Stop-Install {
    param([string]$Message, [int]$Code)
    [Console]::Error.WriteLine($Message)
    exit $Code
}

function Get-Utf8Text {
    param([string]$Base64)
    return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($Base64))
}

try {
    Get-Command -Name $UvCommand -ErrorAction Stop | Out-Null
    & $UvCommand --version | Out-Null
}
catch {
    Stop-Install (Get-Utf8Text "5a6J6KOF5aSx6LSl77ya5pyq5om+5Yiw5Y+v55So55qEIHV244CC") 21
}

if (-not [string]::IsNullOrWhiteSpace($Proxy)) {
    $ProxyUri = $null
    if (-not [Uri]::TryCreate($Proxy, [UriKind]::Absolute, [ref]$ProxyUri) -or
        $ProxyUri.Scheme -notin @("http", "https", "socks5")) {
        Stop-Install (Get-Utf8Text "5a6J6KOF5aSx6LSl77ya5Luj55CG5Zyw5Z2A5b+F6aG75piv5a6M5pW055qEIGh0dHDjgIFodHRwcyDmiJYgc29ja3M1IFVSTOOAgg==") 22
    }
    $env:HTTP_PROXY = $Proxy
    $env:HTTPS_PROXY = $Proxy
    $env:ALL_PROXY = $Proxy
    Write-Output ((Get-Utf8Text "5pys5qyh5a6J6KOF5L2/55So5Luj55CG77ya") + $Proxy)
}

$Extra = "kronos"
if ($Mode -eq "CUDA") {
    try {
        $GpuIdentity = @(& $NvidiaSmiCommand --query-gpu=name,driver_version --format=csv,noheader 2>$null)
        $NvidiaExit = $LASTEXITCODE
    }
    catch {
        Stop-Install (Get-Utf8Text "5a6J6KOF5aSx6LSl77ya5pyq5qOA5rWL5YiwIE5WSURJQSDpqbHliqjvvJvor7fkvb/nlKggLU1vZGUgQ1BVIOaIluWFi+WuieijhempseWKqOOAgg==") 23
    }
    if ($NvidiaExit -ne 0 -or $GpuIdentity.Count -eq 0) {
        Stop-Install (Get-Utf8Text "5a6J6KOF5aSx6LSl77yaTlZJRElBIOmpseWKqOajgOafpeaCquaJgOacqumAmui/h++8jOivt+S9v+eUqCAtTW9kZSBDUFUg5oiW5YWI5pu05paw6amx5Yqo44CC") 23
    }
    Write-Output ((Get-Utf8Text "5qOA5rWL5YiwIE5WSURJQSBHUFXvvJo=") + $GpuIdentity[0])
    $Extra = "kronos-cuda"
}

$SyncArguments = @("sync", "--extra", $Extra)
if ($IncludeDev) {
    $SyncArguments += @("--extra", "dev")
}
$env:UV_HTTP_TIMEOUT = "600"
$env:UV_LOCK_TIMEOUT = "600"

Write-Output ((Get-Utf8Text "5byA5aeL5a6J6KOFIA==") + $Extra + (Get-Utf8Text "77yb5aSn5paH5Lu25LiL6L295pyf6Ze05Y+v6IO95pqC5pe25rKh5pyJ6L+b5bqm6L6T5Ye644CC"))
& $UvCommand @SyncArguments
$SyncExit = $LASTEXITCODE
if ($SyncExit -ne 0) {
    [Console]::Error.WriteLine(
        (Get-Utf8Text "5a6J6KOF5aSx6LSl77yadXYgc3luYyDor5Tlm54g") +
        $SyncExit +
        (Get-Utf8Text "44CC6YeN5paw5omn6KGM5ZCM5LiA5ZG95Luk5Y+v5aSN55So57yT5a2Y44CC")
    )
    exit $SyncExit
}

$CheckCode = @'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
'@
& $UvCommand run python -c $CheckCode
$CheckExit = $LASTEXITCODE
if ($CheckExit -ne 0) {
    Stop-Install (Get-Utf8Text "5a6J6KOF5a6M5oiQ77yM5L2GIFB5VG9yY2gg6Ieq5qOA5aSx6LSl44CC") 24
}

if ($Mode -eq "CUDA") {
    & $UvCommand run python -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Stop-Install (Get-Utf8Text "Q1VEQSDniYjlt7LlrablronvvIzkuYYgUHlUb3JjaCDml6Dms5Xkvb/nlKggR1BV77yb6K+35qOA5p+l6amx5Yqo5ZCO6YeN6K+V44CC") 25
    }
}

Write-Output ("Kronos " + $Mode + (Get-Utf8Text "IOeOr+Wig+WuieijheW5tumqjOivgeWujOaIkOOAgg=="))
