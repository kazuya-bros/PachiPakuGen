param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Downloader,
    [Parameter(Mandatory = $true)][ValidateSet("low-vram", "standard")][string]$Profile,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [Parameter(Mandatory = $true)][string]$Manifest
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "0"

try {
    $Host.UI.RawUI.WindowTitle = "PachiPakuGen - See-Through model download"
} catch {
    # Window title is cosmetic. Continue on hosts without RawUI support.
}

Write-Host "PachiPakuGen See-Through model pre-download" -ForegroundColor Cyan
Write-Host "Profile: $Profile"
Write-Host "The model is downloaded outside the app so byte progress remains visible."
Write-Host "Close this window to pause. Start it again to resume from the partial file."
Write-Host ""

$exitCode = 1
try {
    & $Python $Downloader `
        --profile $Profile `
        --requirements $Requirements `
        --manifest $Manifest
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host $_ -ForegroundColor Red
    $exitCode = 1
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Model pre-download and verification completed." -ForegroundColor Green
    Write-Host "Return to PachiPakuGen and click '環境を再確認'."
    Start-Sleep -Seconds 5
} else {
    Write-Host "Model pre-download did not complete (exit code: $exitCode)." -ForegroundColor Red
    Write-Host "Partial files were kept. Launch the downloader again to resume."
    [void](Read-Host "Press Enter to close")
}

exit $exitCode
