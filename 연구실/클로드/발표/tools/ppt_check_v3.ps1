# PowerPoint로 pptx를 PNG로 뽑고 장마다 깨짐·글 양을 검사합니다 (연구실 클로드, 2026-11-03).
# v3: v2 + '화면 글 합계'(제목 줄·쪽수·바닥선 아래 각주를 뺀 모든 글 = 본문 글상자 + 표 칸 + 캡션) 열과 장 제목 열. G01 R01 검수 1번.
# 사용: powershell -File ppt_check_v2.ps1 -Pptx <파일> -OutDir <폴더> [-Width 1600] [-BottomIn 6.88]
# 검사: (1) 글상자 넘침(글 높이 > 상자 높이+4pt) (2) 장 밖 (3) 바닥선(BottomIn) 아래로 내려온 도형(각주·쪽수 제외)
#       (4) 글상자·표·차트·그림끼리 겹침(선·배경 도형 제외, 2pt 이상) (5) 장별 글자 수: 본문 글상자(13pt 이상) / 표 / 작은 글(12pt 이하) / 노트
param([Parameter(Mandatory = $true)][string]$Pptx, [Parameter(Mandatory = $true)][string]$OutDir, [int]$Width = 1600, [double]$BottomIn = 6.88)
$ErrorActionPreference = "Stop"
$Pptx = (Resolve-Path $Pptx).Path
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$bottom = $BottomIn * 72
$app = New-Object -ComObject PowerPoint.Application
$rows = @(); $issues = @()
function Box($sh) { $l=[double]$sh.Left; $t=[double]$sh.Top; [pscustomobject]@{ L = $l; T = $t; R = $l + [double]$sh.Width; B = $t + [double]$sh.Height; N = [string]$sh.Name } }
try {
    $pres = $app.Presentations.Open($Pptx, $true, $false, $false)
    $slideW = [double]$pres.PageSetup.SlideWidth; $slideH = [double]$pres.PageSetup.SlideHeight
    $h = [int]($Width * $slideH / $slideW)
    foreach ($s in $pres.Slides) {
        $i = [int]$s.SlideIndex
        $s.Export((Join-Path $OutDir ("slide_{0:D2}.png" -f $i)), "PNG", $Width, $h)
        $body = 0; $tab = 0; $small = 0; $boxes = @(); $ttl = ""
        foreach ($sh in $s.Shapes) {
            $isLine = ([int]$sh.Type -eq 9) -or ([double]$sh.Width -lt 1) -or ([double]$sh.Height -lt 1)
            if ($sh.HasTable) {
                $tb = $sh.Table
                for ($r = 1; $r -le $tb.Rows.Count; $r++) { for ($c = 1; $c -le $tb.Columns.Count; $c++) { $tab += ($tb.Cell($r, $c).Shape.TextFrame.TextRange.Text -replace "\s", "").Length } }
                $boxes += ,(Box $sh)
            } elseif ($sh.HasChart) { $boxes += ,(Box $sh) }
            elseif ($sh.HasTextFrame -and $sh.TextFrame.HasText) {
                $tr = $sh.TextFrame.TextRange; $len = ($tr.Text -replace "\s", "").Length
                $isFoot = $sh.Top -ge $bottom - 2
                $isTitle = ([double]$sh.Top -lt 79) -and ([double]$sh.Height -lt 45)
                if ($isTitle -and -not $ttl -and $len -gt 3) { $ttl = ($tr.Text -replace "\s+", " ") }
                if ($isFoot -or $isTitle) { }
                elseif ($tr.Font.Size -le 12) { $small += $len } else { $body += $len }
                $bh = $sh.TextFrame2.TextRange.BoundHeight
                if ($bh -gt $sh.Height + 4) { $issues += "{0:D2} 넘침 {1}: 글 {2:N0}pt > 상자 {3:N0}pt '{4}'" -f $i, $sh.Name, $bh, $sh.Height, ($tr.Text -replace "\s+", " ").Substring(0, [Math]::Min(25, $tr.Text.Length)) }
                if (-not $isFoot) { $boxes += ,(Box $sh) }
            }
            if (-not $isLine) {
                $L=[double]$sh.Left; $Tp=[double]$sh.Top; $Bt=$Tp+[double]$sh.Height; if ($L -lt -1 -or $Tp -lt -1 -or ($L + [double]$sh.Width) -gt $slideW + 1 -or $Bt -gt $slideH + 1) { $issues += "{0:D2} 장 밖 {1}" -f $i, $sh.Name }
                if ($Tp -lt $bottom - 2 -and $Bt -gt $bottom + 2) { $issues += "{0:D2} 바닥선 넘음 {1}: 아래 끝 {2:N2}in" -f $i, $sh.Name, ($Bt / 72) }
            }
        }
        for ($a = 0; $a -lt $boxes.Count; $a++) { for ($b = $a + 1; $b -lt $boxes.Count; $b++) {
            $p = $boxes[$a]; $q = $boxes[$b]
            $ox = [Math]::Min($p.R, $q.R) - [Math]::Max($p.L, $q.L); $oy = [Math]::Min($p.B, $q.B) - [Math]::Max($p.T, $q.T)
            if ($ox -gt 2 -and $oy -gt 2) { $issues += "{0:D2} 겹침 {1} × {2} ({3:N0}×{4:N0}pt)" -f $i, $p.N, $q.N, $ox, $oy }
        } }
        $notes = 0; try { $notes = ($s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text -replace "\s", "").Length } catch {}
        $rows += "{0:D2}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}" -f $i, ($body + $tab + $small), $body, $tab, $small, $notes, $ttl
    }
    $fonts = @(); foreach ($f in $pres.Fonts) { $fonts += $f.Name }
    $pres.Close()
} finally { $app.Quit() }
$out = @("장`t화면글합계`t본문글상자`t표글자`t캡션(12pt이하)`t노트글자`t제목") + $rows + @("", "문제 " + $issues.Count + "건") + $issues + @("", "글꼴: " + ($fonts -join ", "))
$out | Set-Content -Encoding utf8 (Join-Path $OutDir "검사_v3.txt")
$out
