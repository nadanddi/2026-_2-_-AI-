# PowerPoint로 pptx의 모든 장을 PNG로 뽑고, 장마다 글꼴 대체·글상자 넘침을 검사합니다 (연구실 클로드, 2026-11-03).
# 사용: powershell -File ppt_export_v1.ps1 -Pptx <파일> -OutDir <폴더> [-Width 1600]
param([Parameter(Mandatory = $true)][string]$Pptx, [Parameter(Mandatory = $true)][string]$OutDir, [int]$Width = 1600)
$ErrorActionPreference = "Stop"
$Pptx = (Resolve-Path $Pptx).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($Pptx, $true, $false, $false)  # 읽기 전용, 창 없음
    $h = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    $report = @()
    foreach ($s in $pres.Slides) {
        $i = $s.SlideIndex
        $s.Export((Join-Path $OutDir ("slide_{0:D2}.png" -f $i)), "PNG", $Width, $h)
        foreach ($sh in $s.Shapes) {
            if ($sh.HasTextFrame -and $sh.TextFrame.HasText) {
                $tf = $sh.TextFrame2
                $bh = $tf.TextRange.BoundHeight
                if ($bh -gt $sh.Height + 4) { $report += "{0:D2} 넘침 {1} (글 높이 {2:N0}pt > 상자 {3:N0}pt): {4}" -f $i, $sh.Name, $bh, $sh.Height, ($sh.TextFrame.TextRange.Text -replace "\s+", " ").Substring(0, [Math]::Min(30, $sh.TextFrame.TextRange.Text.Length)) }
                if (($sh.Left + $sh.Width) -gt $pres.PageSetup.SlideWidth + 1 -or ($sh.Top + $sh.Height) -gt $pres.PageSetup.SlideHeight + 1) { $report += "{0:D2} 장 밖 {1}" -f $i, $sh.Name }
            }
        }
    }
    $fonts = @(); foreach ($f in $pres.Fonts) { $fonts += $f.Name }
    $report += "글꼴: " + ($fonts -join ", ")
    $pres.Close()
} finally { $app.Quit() }
$report | Set-Content -Encoding utf8 (Join-Path $OutDir "검사.txt")
$report
