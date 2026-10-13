$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location -LiteralPath $root
$logRoot = Join-Path $root 'data\ablation_logs'
$statusPath = Join-Path $logRoot 'run_status.json'
$finalLog = Join-Path $logRoot 'finalization.log'

while ($true) {
    if (Test-Path -LiteralPath $statusPath) {
        try {
            $status = @(Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json)
            if ($status.Count -ge 3) { break }
        } catch {}
    }
    Start-Sleep -Seconds 60
}

"[$((Get-Date).ToString('o'))] Finalization started" | Set-Content -LiteralPath $finalLog -Encoding utf8
& .\.venv\Scripts\python.exe scripts\ablation\report_simple_unet_ablation.py 2>&1 |
    Tee-Object -FilePath $finalLog -Append
$reportExit = $LASTEXITCODE
& .\.venv\Scripts\python.exe -m pytest -q 2>&1 | Tee-Object -FilePath $finalLog -Append
$pytestExit = $LASTEXITCODE
& git diff --check 2>&1 | Tee-Object -FilePath $finalLog -Append
$diffCheckExit = $LASTEXITCODE

$protected = @(
    'data\models\simple_unet_V0\simple_unet_airfoil.pt',
    'data\models\simple_unet_improved\simple_unet_airfoil.pt',
    'data\models\simple_unet\fixed_test_split.json'
)
$after = @{}
foreach ($path in $protected) {
    $after[$path] = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower()
}
$after | ConvertTo-Json | Set-Content -LiteralPath "$logRoot\protected_hashes_after.json" -Encoding utf8
$before = Get-Content -Raw -LiteralPath "$logRoot\protected_hashes_before.json" | ConvertFrom-Json
$integrity = @{}
foreach ($path in $protected) {
    $integrity[$path] = ($before.PSObject.Properties[$path].Value -eq $after[$path])
}
[pscustomobject]@{
    completed = (Get-Date).ToString('o')
    report_exit = $reportExit
    pytest_exit = $pytestExit
    git_diff_check_exit = $diffCheckExit
    protected_artifacts_unchanged = $integrity
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$logRoot\final_status.json" -Encoding utf8
