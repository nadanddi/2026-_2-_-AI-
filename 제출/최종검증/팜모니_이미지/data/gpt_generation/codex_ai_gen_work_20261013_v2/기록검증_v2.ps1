$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$sourceRoot = 'C:\work\farmai\연구실\클로드\local\ai_gen_work\원본2'
$sourceListPath = 'C:\work\farmai\연구실\클로드\local\ai_gen_work\원본2_목록.csv'
$outputNames = @('기록_v2.csv','manifest_v2.csv','파일검산_v2.json','검증_v2.json')
foreach ($name in $outputNames) {
    if (Test-Path -LiteralPath (Join-Path $taskRoot $name)) { throw "Existing output: $name" }
}
Add-Type -AssemblyName System.Drawing
$records = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot '기록') -File -Filter '*.json' |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json } |
    Sort-Object n)
$sourceList = @(Import-Csv -LiteralPath $sourceListPath -Encoding UTF8)
$sourceMap = @{}
foreach ($entry in $sourceList) {
    if ($sourceMap.ContainsKey($entry.'파일명')) { throw 'Duplicate source list filename' }
    $sourceMap[$entry.'파일명'] = $entry
}
$pngFiles = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot '결과_GPT') -Recurse -File -Filter '*.png')
if ($records.Count -ne 150 -or $pngFiles.Count -ne 150 -or $sourceList.Count -ne 150) { throw 'Expected 150 records, outputs and sources' }
if (@($records.source | Sort-Object -Unique).Count -ne 150) { throw 'Duplicate source use' }
if (@($records.result | Sort-Object -Unique).Count -ne 150) { throw 'Duplicate result path' }
if (@($records.generated_path | Sort-Object -Unique).Count -ne 150) { throw 'Duplicate service output' }
$manifest = [System.Collections.Generic.List[object]]::new()
$logRows = [System.Collections.Generic.List[object]]::new()
$details = [System.Collections.Generic.List[object]]::new()
foreach ($record in $records) {
    $expectedId = 'src2_{0:D3}' -f [int]$record.n
    $expectedType = if ([int]$record.n -le 100) { '3_부분생성' } else { '4_부분스타일' }
    if ($record.id -ne $expectedId -or $record.source -ne "$expectedId.jpg" -or $record.type -ne $expectedType) { throw "Assignment mismatch: $expectedId" }
    if ($record.tool -ne 'GPT_builtin_imagegen') { throw 'Unexpected generator' }
    $sourceEntry = $sourceMap[$record.source]
    if ($null -eq $sourceEntry -or [int]$sourceEntry.'원본번호' -ne [int]$record.n) { throw 'Source lineage mismatch' }
    $relativeResult = "결과_GPT/$($record.type)/$($record.filename)"
    $expectedResult = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $relativeResult))
    $actualResult = [System.IO.Path]::GetFullPath($record.result)
    if ($actualResult -cne $expectedResult) { throw "Result path mismatch: $expectedId" }
    $sourcePath = Join-Path $sourceRoot $record.source
    $promptRecord = Get-Content -LiteralPath (Join-Path $taskRoot "사전프롬프트/$expectedId.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($promptRecord.prompt -cne $record.prompt) { throw "Prompt snapshot mismatch: $expectedId" }
    $required = @('원본과 같은 가로세로 비율','사람·얼굴·새 글씨·워터마크 없음, 결과 한 장만')
    if ($expectedType -eq '3_부분생성') {
        $required += @('그 작은 영역만 새로 그리고 나머지 구도·사물·잎·시설·바닥·조명·색감은 그대로 보존','원래 사진과 구분 안 되게 사진처럼 사실적으로')
    } else {
        $required += @('나머지는 사진 그대로, 사진 전체를 다시 그리지 말 것')
    }
    foreach ($phrase in $required) {
        if (-not $record.prompt.Contains($phrase)) { throw "Required prompt text missing: $expectedId" }
    }
    $outputHash = (Get-FileHash -LiteralPath $actualResult -Algorithm SHA256).Hash
    $rawHash = (Get-FileHash -LiteralPath $record.generated_path -Algorithm SHA256).Hash
    $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
    if ($outputHash -cne $rawHash) { throw "Raw copy hash mismatch: $expectedId" }
    $image = [System.Drawing.Image]::FromFile($actualResult)
    try {
        if ($image.RawFormat.Guid -ne [System.Drawing.Imaging.ImageFormat]::Png.Guid) { throw 'Result is not PNG' }
        $width = $image.Width
        $height = $image.Height
        $decodedBitmap = [System.Drawing.Bitmap]::new($image)
        try { $null = $decodedBitmap.GetPixel($width - 1, $height - 1) }
        finally { $decodedBitmap.Dispose() }
    } finally { $image.Dispose() }
    $sourceImage = [System.Drawing.Image]::FromFile($sourcePath)
    try { $sourceWidth = $sourceImage.Width; $sourceHeight = $sourceImage.Height }
    finally { $sourceImage.Dispose() }
    $bytes = (Get-Item -LiteralPath $actualResult).Length
    $aspectRelativeError = [math]::Abs(($width / $height) / ($sourceWidth / $sourceHeight) - 1.0)
    $manifest.Add([pscustomobject][ordered]@{
        id=$record.id; type=$record.type; file=$relativeResult
        width=$width; height=$height; bytes=$bytes; sha256=$outputHash
        source=$record.source; source_sha256=$sourceHash
        farm=$sourceEntry.'농장코드'; aihub_source=$sourceEntry.'AI허브_원천'
        raw_copy_identical=$true; prompt=$record.prompt
    })
    $logRows.Add([pscustomobject][ordered]@{
        '도구'='GPT'; '유형'=$record.type; '결과파일'=$relativeResult
        '원본파일'=$record.source; '프롬프트'=$record.prompt
    })
    $details.Add([pscustomobject][ordered]@{
        id=$record.id; source_path=$sourcePath; result_path=$actualResult
        generated_path=$record.generated_path; raw_sha256=$rawHash
        output_sha256=$outputHash; source_sha256=$sourceHash
        source_width=$sourceWidth; source_height=$sourceHeight
        width=$width; height=$height; bytes=$bytes
        aspect_ratio_relative_error=$aspectRelativeError; decode_ok=$true
    })
}
$expectedIds = @(1..150 | ForEach-Object { 'src2_{0:D3}' -f $_ })
if (Compare-Object -ReferenceObject $expectedIds -DifferenceObject @($records.id)) { throw 'Missing/extra IDs' }
$generated = @($records | Where-Object type -eq '3_부분생성')
$styled = @($records | Where-Object type -eq '4_부분스타일')
$smallCount = @($generated | Where-Object size -eq '잎 1~2장').Count
if ($generated.Count -ne 100 -or $styled.Count -ne 50 -or $smallCount -lt 50) { throw 'Type or planned size count mismatch' }
$actionCounts = @($generated | Group-Object action | Select-Object Name,Count)
$intensityCounts = @($styled | Group-Object intensity | Select-Object Name,Count)
$styleCounts = @($styled | Group-Object style | Select-Object Name,Count)
if ((@($actionCounts.Count | Sort-Object) -join ',') -ne '33,33,34') { throw 'Action balance mismatch' }
if ((@($intensityCounts.Count | Sort-Object) -join ',') -ne '25,25') { throw 'Intensity balance mismatch' }
if ($styleCounts.Count -ne 5 -or @($styleCounts | Where-Object Count -ne 10).Count -ne 0) { throw 'Style balance mismatch' }
if (@($manifest.sha256 | Sort-Object -Unique).Count -ne 150) { throw 'Duplicate PNG content' }
$manifest | Export-Csv -LiteralPath (Join-Path $taskRoot 'manifest_v2.csv') -NoTypeInformation -Encoding utf8BOM
$logRows | Export-Csv -LiteralPath (Join-Path $taskRoot '기록_v2.csv') -NoTypeInformation -Encoding utf8BOM
$manifestReloaded = @(Import-Csv -LiteralPath (Join-Path $taskRoot 'manifest_v2.csv') -Encoding UTF8)
$logReloaded = @(Import-Csv -LiteralPath (Join-Path $taskRoot '기록_v2.csv') -Encoding UTF8)
$referenceManifest = @(Import-Csv -LiteralPath 'C:\work\farmai\연구실\코덱스\local\ai_gen_work_20261013_v1\manifest_v1.csv' -Encoding UTF8)
$referenceLog = @(Import-Csv -LiteralPath 'C:\work\farmai\연구실\코덱스\local\ai_gen_work_20261013_v1\기록_v1.csv' -Encoding UTF8)
if (($manifestReloaded[0].PSObject.Properties.Name -join '|') -cne ($referenceManifest[0].PSObject.Properties.Name -join '|')) { throw 'Manifest columns differ from v1' }
if (($logReloaded[0].PSObject.Properties.Name -join '|') -cne ($referenceLog[0].PSObject.Properties.Name -join '|')) { throw 'Log columns differ from v1' }
if ($manifestReloaded.Count -ne 150 -or $logReloaded.Count -ne 150) { throw 'CSV reload count mismatch' }
for ($index = 0; $index -lt 150; $index++) {
    $mr = $manifestReloaded[$index]
    $lr = $logReloaded[$index]
    $rr = $records[$index]
    if ($mr.prompt -cne $rr.prompt -or $lr.'프롬프트' -cne $rr.prompt -or $mr.source -cne $rr.source -or $lr.'원본파일' -cne $rr.source -or $mr.file -cne $lr.'결과파일') { throw 'CSV roundtrip linkage mismatch' }
    if (-not (Test-Path -LiteralPath (Join-Path $taskRoot $mr.file))) { throw 'CSV points to missing PNG' }
    if ($mr.aihub_source -cne $sourceMap[$rr.source].'AI허브_원천') { throw 'AIHub lineage mismatch' }
}
$verification = [pscustomobject][ordered]@{
    total=150; by_type=@($manifest | Group-Object type | Select-Object Name,Count)
    unique_sources=@($manifest.source | Sort-Object -Unique).Count
    unique_source_sha256=@($manifest.source_sha256 | Sort-Object -Unique).Count
    by_farm=@($manifest | Group-Object farm | Sort-Object Name | Select-Object Name,Count)
    bytes=($manifest | Measure-Object bytes -Sum).Sum
    all_raw_copies_identical=$true; raw_copy_checks=150
    all_decode_ok=$true; decode_checks=150
    unique_output_sha256=150; csv_roundtrip_linkage_checks=150
    same_columns_as_v1=$true; prompt_snapshot_checks=150
    required_prompt_phrase_checks=150; source_assignment_checks=150
    generation_action_counts=$actionCounts
    generation_requested_small_edit_count=$smallCount
    generation_requested_size_counts=@($generated | Group-Object size | Select-Object Name,Count)
    generation_requested_position_counts=@($generated | Group-Object area | Select-Object Name,Count)
    style_requested_style_counts=$styleCounts; style_requested_intensity_counts=$intensityCounts
    max_aspect_ratio_relative_error=($details | Measure-Object aspect_ratio_relative_error -Maximum).Maximum
    failed_or_refused=0; postprocessing='none'
    source_list_sha256=(Get-FileHash -LiteralPath $sourceListPath -Algorithm SHA256).Hash
    request_sha256=(Get-FileHash -LiteralPath 'C:\work\farmai\연구실\클로드\local\ai_gen_work\GPT_추가요청_v2.md' -Algorithm SHA256).Hash
    generation_tool='GPT_builtin_imagegen'; model_version='unknown'; service_seed='unknown'
    semantic_pixel_invariance_verified=$false
}
$details | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskRoot '파일검산_v2.json') -Encoding utf8
$verification | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskRoot '검증_v2.json') -Encoding utf8
$verification | Select-Object total,unique_sources,unique_source_sha256,bytes,all_raw_copies_identical,all_decode_ok,unique_output_sha256,csv_roundtrip_linkage_checks,generation_requested_small_edit_count,max_aspect_ratio_relative_error | ConvertTo-Json -Depth 5

