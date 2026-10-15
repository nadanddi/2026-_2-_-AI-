$ErrorActionPreference = 'Continue'
$taskRemote = 'storage/nadan/codex/JOB001_SZ2_20261014_v1'
$taskDrive = 'G:\내 드라이브\farmai_sync\gpu_jobs'
$taskResult = Join-Path $taskDrive 'JOB001_SZ2_계절지표_실제파이프라인\results'
while ($true) {
    $taskStatus = (& ssh -o BatchMode=yes -o ConnectTimeout=15 ADILAB.iptime.org "cat ~/$taskRemote/STATUS.txt" 2>$null | Out-String).Trim()
    if ($taskStatus -eq 'ERROR') {
        $taskMismatch = (& ssh -o BatchMode=yes ADILAB.iptime.org "grep -c 'exit_code=0 checkpoint_count=62 expected=66' ~/$taskRemote/results/ERROR.txt" 2>$null | Out-String).Trim()
        if ($taskMismatch -eq '1') { Start-Sleep -Seconds 60; continue }
    }
    if ($taskStatus -in @('COMPLETE', 'ERROR')) {
        New-Item -ItemType Directory -Path $taskResult -Force | Out-Null
        & scp -o BatchMode=yes -r "ADILAB.iptime.org:$taskRemote/results/." $taskResult
        if ($LASTEXITCODE -eq 0) {
            $taskQueue = Join-Path $taskDrive 'QUEUE.md'
            $taskText = Get-Content -LiteralPath $taskQueue -Raw -Encoding utf8
            $taskLabel = if ($taskStatus -eq 'COMPLETE') { '완료' } else { '오류' }
            $taskTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $taskText = [regex]::Replace($taskText, '(?m)^(\| JOB001 \|)[^|]*(\|)', ('$1 ' + $taskLabel + ' (' + $taskTime + ', 연구실 서버 GPU0) $2'))
            Set-Content -LiteralPath $taskQueue -Value $taskText -Encoding utf8
            break
        }
    }
    Start-Sleep -Seconds 60
}
