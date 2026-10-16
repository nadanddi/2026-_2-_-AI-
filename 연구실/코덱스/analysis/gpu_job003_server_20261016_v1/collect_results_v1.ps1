$ErrorActionPreference = 'Stop'
$taskRemote = 'storage/nadan/codex/JOB003_FT2_20261016_v1'
$taskDrive = 'G:\내 드라이브\farmai_sync\gpu_jobs'
$taskResult = Join-Path $taskDrive 'JOB003_FT2_과거만학습_온도\results'
$taskLog = 'C:\work\farmai\연구실\코덱스\analysis\gpu_job003_server_20261016_v1\collector_v1.log'
while ($true) {
    $taskStatus = (& ssh -o BatchMode=yes -o ConnectTimeout=15 ADILAB.iptime.org "cat ~/$taskRemote/STATUS.txt" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and $taskStatus -in @('COMPLETE', 'ERROR')) {
        New-Item -ItemType Directory -Path $taskResult -Force | Out-Null
        & scp -o BatchMode=yes -r "ADILAB.iptime.org:$taskRemote/results/." $taskResult 2>&1 | Out-File -LiteralPath $taskLog -Append -Encoding utf8
        if ($LASTEXITCODE -eq 0) {
            $taskQueue = Join-Path $taskDrive 'QUEUE.md'
            $taskQueueText = Get-Content -LiteralPath $taskQueue -Raw -Encoding utf8
            $taskLabel = if ($taskStatus -eq 'COMPLETE') { '완료' } else { '오류' }
            $taskTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $taskQueueText = [regex]::Replace($taskQueueText, '(?m)^(\| JOB003 \|)[^|]*(\|)', ('$1 ' + $taskLabel + ' (' + $taskTime + ', 연구실 서버 GPU0) $2'))
            Set-Content -LiteralPath $taskQueue -Value $taskQueueText -Encoding utf8
            "$taskTime $taskStatus results copied" | Out-File -LiteralPath $taskLog -Append -Encoding utf8
            break
        }
    }
    Start-Sleep -Seconds 60
}

