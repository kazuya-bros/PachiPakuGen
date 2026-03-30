# Build PachiPakuGen Tauri app (dev mode - fast compile)
# Run this from PowerShell to ensure MSVC link.exe is used

Set-Location "E:\develop\PachiPakuGen\PachiPakuGen-app\src-tauri"

# Ensure cargo is in PATH
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# cargo build (debug)
cargo build 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Build successful! ===" -ForegroundColor Green
    Write-Host "EXE: target\debug\pachipakugen-app.exe"
} else {
    Write-Host "`n=== Build failed ===" -ForegroundColor Red
}
