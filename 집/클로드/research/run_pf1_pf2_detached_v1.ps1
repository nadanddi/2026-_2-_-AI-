# Detached runner (2026-10-07 집 클로드): finish PF1 (2 GPU workers, resumes from checkpoints), aggregate,
# then PF2 (2 GPU workers) and aggregate.  Logs: pf1_*.log / pf2_*.log in this folder; status in run_pf_status.txt.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = ""
function Stamp($m) { Add-Content -LiteralPath "run_pf_status.txt" -Value ("{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $m) -Encoding utf8 }
Stamp "start PF1 workers"
$p = @()
foreach ($k in 0, 1) { $p += Start-Process -FilePath "py" -ArgumentList "-3.12", "-u", "run_parallel_folds_v1.py", "ec3_PF1_tabpfn_upgrade_v1.py", "$k", "2" -RedirectStandardOutput "pf1_worker${k}_b.log" -RedirectStandardError "pf1_worker${k}_b.err" -NoNewWindow -PassThru }
$p | Wait-Process
Stamp "PF1 workers done; aggregate"
Start-Process -FilePath "py" -ArgumentList "-3.12", "-u", "ec3_PF1_tabpfn_upgrade_v1.py" -RedirectStandardOutput "ec3_PF1_tabpfn_upgrade_v1.log" -RedirectStandardError "ec3_PF1_tabpfn_upgrade_v1.err" -NoNewWindow -Wait
Stamp "PF1 aggregated; start PF2 workers"
$p = @()
foreach ($k in 0, 1) { $p += Start-Process -FilePath "py" -ArgumentList "-3.12", "-u", "run_parallel_folds_v1.py", "ec3_PF2_anchor_difference_features_v1.py", "$k", "2" -RedirectStandardOutput "pf2_worker$k.log" -RedirectStandardError "pf2_worker$k.err" -NoNewWindow -PassThru }
$p | Wait-Process
Stamp "PF2 workers done; aggregate"
Start-Process -FilePath "py" -ArgumentList "-3.12", "-u", "ec3_PF2_anchor_difference_features_v1.py" -RedirectStandardOutput "ec3_PF2_anchor_difference_features_v1.log" -RedirectStandardError "ec3_PF2_anchor_difference_features_v1.err" -NoNewWindow -Wait
Stamp "ALL DONE"
