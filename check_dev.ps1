# Check PachiPakuGen Tauri app (dev mode - fast compile check only)
Set-Location "E:\develop\PachiPakuGen\PachiPakuGen-app\src-tauri"

# Ensure cargo is in PATH
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# cargo check (fast, no codegen)
cargo check 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Check passed! ===" -ForegroundColor Green
} else {
    Write-Host "`n=== Check failed ===" -ForegroundColor Red
}
