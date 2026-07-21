# Build PachiPakuGen Tauri app (dev mode - fast compile)
# Run this from PowerShell to ensure MSVC link.exe is used

$repoRoot = Split-Path $PSScriptRoot -Parent
$tauriRoot = Join-Path $repoRoot "src-tauri"
Push-Location $tauriRoot

# Ensure cargo is in PATH
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# cargo build (debug)
$exitCode = 0
try {
    cargo build 2>&1
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -eq 0) {
    Write-Host "`n=== Build successful! ===" -ForegroundColor Green
    Write-Host "EXE: target\debug\pachipakugen-app.exe"
} else {
    Write-Host "`n=== Build failed ===" -ForegroundColor Red
}

exit $exitCode
