# Check PachiPakuGen Tauri app (dev mode - fast compile check only)
$repoRoot = $PSScriptRoot
$tauriRoot = Join-Path $repoRoot "src-tauri"
Push-Location $tauriRoot

# Ensure cargo is in PATH
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# cargo check (fast, no codegen)
$exitCode = 0
try {
    cargo check 2>&1
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -eq 0) {
    Write-Host "`n=== Check passed! ===" -ForegroundColor Green
} else {
    Write-Host "`n=== Check failed ===" -ForegroundColor Red
}

exit $exitCode
