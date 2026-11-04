# 작업·검수 반복에서 상대 파일이 생겼는지 확인합니다 (규칙.md).
#  -Role claude : 가장 최근 작업 파일과 짝인 검수 파일을 기다림 (보통 -FirstWaitSec 300)
#  -Role gpt    : 짝이 되는 검수 파일이 없는 작업 파일, 또는 끝.md를 기다림
# 찾으면 파일 경로를 출력하고 종료 코드 0, 시간 초과면 2.
param(
    [Parameter(Mandatory = $true)][ValidateSet("claude", "gpt")][string]$Role,
    [int]$FirstWaitSec = 0,
    [int]$EverySec = 120,
    [int]$MaxMinutes = 240
)
$ErrorActionPreference = "Stop"
$Dir = Join-Path $PSScriptRoot "라운드"
if (-not (Test-Path $Dir)) { New-Item -ItemType Directory $Dir | Out-Null }

function Find-Target {
    $work = Get-ChildItem $Dir -Filter "G*_R*_작업.md" -File | Sort-Object Name
    if ($Role -eq "claude") {
        $last = $work | Select-Object -Last 1
        if (-not $last) { return $null }
        $review = Join-Path $Dir ($last.Name -replace "_작업\.md$", "_검수.md")
        if (Test-Path $review) { return $review }
        return $null
    }
    foreach ($w in $work) {
        $review = Join-Path $Dir ($w.Name -replace "_작업\.md$", "_검수.md")
        if (-not (Test-Path $review)) { return $w.FullName }
    }
    $end = Join-Path $Dir "끝.md"
    if (Test-Path $end) { return $end }
    return $null
}

$deadline = (Get-Date).AddMinutes($MaxMinutes)
if ($FirstWaitSec -gt 0) {
    Write-Host ("[{0:HH:mm}] {1}초 기다린 뒤 확인합니다." -f (Get-Date), $FirstWaitSec)
    Start-Sleep -Seconds $FirstWaitSec
}
while ($true) {
    $hit = Find-Target
    if ($hit) { Write-Host ("[{0:HH:mm}] 찾음" -f (Get-Date)); Write-Output $hit; exit 0 }
    if ((Get-Date) -ge $deadline) { Write-Host ("[{0:HH:mm}] {1}분 동안 없음, 종료" -f (Get-Date), $MaxMinutes); exit 2 }
    Write-Host ("[{0:HH:mm}] 없음, {1}초 뒤 다시 확인" -f (Get-Date), $EverySec)
    Start-Sleep -Seconds $EverySec
}
