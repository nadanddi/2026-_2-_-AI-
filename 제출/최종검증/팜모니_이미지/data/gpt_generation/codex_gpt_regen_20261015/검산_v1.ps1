$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$detailPath = Join-Path $taskRoot '파일검산_v1.json'
$summaryPath = Join-Path $taskRoot '검증_v1.json'
if ((Test-Path -LiteralPath $detailPath) -or (Test-Path -LiteralPath $summaryPath)) { throw '검증 산출물 존재. 새 버전으로 실행할 것.' }
$jobs = @(Get-Content -LiteralPath (Join-Path $taskRoot '선정_v1.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
$records = @(Get-Content -LiteralPath (Join-Path $taskRoot '전체호출_v1.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
$plan = Get-Content -LiteralPath (Join-Path $taskRoot '계획_v1.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$mapping = @(Import-Csv -LiteralPath (Join-Path $taskRoot '목록.csv') -Encoding UTF8)
$prefix = @(Import-Csv -LiteralPath $plan.manifest -Encoding UTF8 | Where-Object role -eq 'itrain' | Select-Object -First 150)
$pngs = @(Get-ChildItem -LiteralPath $taskRoot -File -Filter '*.png')
$preFiles = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot '사전프롬프트') -File -Filter '*.json')
$recordFiles = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot '기록') -File -Filter '*.json')
foreach ($pair in @(@('selection',$jobs.Count),@('records',$records.Count),@('csv',$mapping.Count),@('manifest_prefix',$prefix.Count),@('png',$pngs.Count),@('preprompt',$preFiles.Count),@('record_files',$recordFiles.Count))) {
 if ($pair[1] -ne 150) { throw "$($pair[0]) count $($pair[1])" }
}
if (($mapping[0].PSObject.Properties.Name -join ',') -cne '원본파일명,결과파일명') { throw 'CSV headers' }
Add-Type -AssemblyName System.Drawing
$details = @(
for ($i=0; $i -lt 150; $i++) {
 $j=$jobs[$i]; $r=$records[$i]; $c=$mapping[$i]; $m=$prefix[$i]
 $name = '{0:D3}.json' -f ($i+1)
 $p=Get-Content -LiteralPath (Join-Path (Join-Path $taskRoot '사전프롬프트') $name) -Raw -Encoding UTF8 | ConvertFrom-Json
 $diskR=Get-Content -LiteralPath (Join-Path (Join-Path $taskRoot '기록') $name) -Raw -Encoding UTF8 | ConvertFrom-Json
 if ($j.n -ne $i+1 -or $r.n -ne $j.n -or $p.n -ne $j.n -or $diskR.n -ne $j.n) { throw "order $i" }
 if ($m.file -cne $j.source_file -or $m.tar -cne $j.tar -or $j.role -cne 'itrain') { throw "manifest prefix $i" }
 if ($c.'원본파일명' -cne $j.source_file -or $c.'결과파일명' -cne $j.result_file) { throw "csv order $i" }
 if ([IO.Path]::ChangeExtension($j.source_file,'png') -cne $j.result_file) { throw "filename $i" }
 foreach ($field in @('source_file','result_file','source','source_sha256','source_bytes','tar','farm','group','role')) {
  if ($r.$field -cne $j.$field -or $p.$field -cne $j.$field -or $diskR.$field -cne $j.$field) { throw "record/preprompt $i $field" }
 }
 if ($p.prompt -cne $plan.prompt -or $r.prompt -cne $plan.prompt -or $diskR.prompt -cne $plan.prompt -or $r.tool -cne 'GPT_builtin_imagegen') { throw "prompt/tool $i" }
 $sourceItem=Get-Item -LiteralPath $j.source
 $sourceHash=(Get-FileHash -LiteralPath $j.source -Algorithm SHA256).Hash
 if ($sourceHash -cne $j.source_sha256 -or $sourceItem.Length -ne $j.source_bytes) { throw "source changed $i" }
 $out=Join-Path $taskRoot $j.result_file
 if ([IO.Path]::GetFullPath($out) -cne [IO.Path]::GetFullPath($r.result)) { throw "result path $i" }
 $outItem=Get-Item -LiteralPath $out
 $outHash=(Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash
 $rawHash=(Get-FileHash -LiteralPath $r.raw -Algorithm SHA256).Hash
 if ($outHash -cne $r.sha256 -or $outHash -cne $rawHash -or $outHash -cne $diskR.sha256 -or $r.raw -cne $diskR.raw -or $outItem.Length -ne $r.bytes -or $outItem.Length -ne (Get-Item -LiteralPath $r.raw).Length) { throw "copy/hash $i" }
 $img=[Drawing.Bitmap]::FromFile($out)
 try {
  $w=$img.Width; $h=$img.Height
  if ($w -lt 1 -or $h -lt 1 -or $img.RawFormat.Guid -ne [Drawing.Imaging.ImageFormat]::Png.Guid) { throw "PNG format $i" }
  $null=$img.GetPixel([int]($w/2),[int]($h/2))
 } finally { $img.Dispose() }
 [pscustomobject]@{n=$j.n;source_file=$j.source_file;result_file=$j.result_file;source_sha256=$sourceHash;sha256=$outHash;bytes=$outItem.Length;width=$w;height=$h;source_unchanged=$true;raw_copy_equal=$true;manifest_prefix_equal=$true;csv_equal=$true;prompt_record_equal=$true;png_readable=$true}
})
foreach ($field in @('source_file','result_file','sha256','source_sha256')) {
 if (@($details.$field | Sort-Object -Unique).Count -ne 150) { throw "unique $field" }
}
if (@($records.raw | Sort-Object -Unique).Count -ne 150) { throw 'unique raw' }
$summary=[ordered]@{
 verified_at=(Get-Date -Format o); pass=$true; count=150
 checks=@('manifest role=itrain 처음150 file/tar 순서 일치','원본 SHA256/bytes 불변','PNG150 읽기/형식','파일 basename 유지','생성 원출력/저장 PNG/기록 SHA256 일치','목록 CSV2열150행 순서 일치','사전 prompt/개별기록/전체기록150 대응','원본/출력/생성raw150 중복 없음')
 total_png_bytes=($details | Measure-Object bytes -Sum).Sum
 dimensions=@($details | Group-Object width,height | ForEach-Object { [pscustomobject]@{size=$_.Name;count=$_.Count} })
 farms=@($jobs | Group-Object farm | ForEach-Object { [pscustomobject]@{farm=$_.Name;count=$_.Count} })
 first_completed=$records[0].completed_at;last_completed=$records[-1].completed_at
 local_gpu_use_this_task=0;training_evaluation_submission_this_task=0
 verification_scope='파일·순서·출처 무결성 검산. 내용 추가/삭제 0 및 OCR 동일성을 인증하는 검증은 아님.'
 manifest_note=$plan.note
}
$utf8=New-Object Text.UTF8Encoding($true)
[IO.File]::WriteAllText($detailPath,($details | ConvertTo-Json -Depth 8),$utf8)
[IO.File]::WriteAllText($summaryPath,($summary | ConvertTo-Json -Depth 8),$utf8)
$summary | ConvertTo-Json -Depth 8

