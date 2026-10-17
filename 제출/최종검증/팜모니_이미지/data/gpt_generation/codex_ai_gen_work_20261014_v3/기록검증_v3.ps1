param([switch]$WriteCsv)
$ErrorActionPreference = 'Stop'
$batchRoot = 'C:\work\farmai\연구실\코덱스\local\ai_gen_work_20261014_v3'
$sourceRoot = 'C:\work\farmai\연구실\클로드\local\ai_gen_work\원본3'
$sourceList = 'C:\work\farmai\연구실\클로드\local\ai_gen_work\원본3_목록.csv'
$v2Root = 'C:\work\farmai\연구실\코덱스\local\ai_gen_work_20261013_v2'
$requiredPhrase = '사람·얼굴·새 글씨·워터마크 없음, 결과 한 장만'
function Assert-True($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Read-Dimensions([string]$Path, [switch]$Png) {
    $taskImage = [System.Drawing.Image]::FromFile($Path)
    try {
        if ($Png) {
            Assert-True ($taskImage.RawFormat.Guid -eq [System.Drawing.Imaging.ImageFormat]::Png.Guid) "PNG format: $Path"
            $taskBitmap = [System.Drawing.Bitmap]::new($taskImage)
            try { $null = $taskBitmap.GetPixel($taskBitmap.Width-1, $taskBitmap.Height-1) }
            finally { $taskBitmap.Dispose() }
        }
        Assert-True ($taskImage.Width -gt 0 -and $taskImage.Height -gt 0) "Dimensions: $Path"
        [pscustomobject]@{ width=$taskImage.Width; height=$taskImage.Height }
    } finally { $taskImage.Dispose() }
}
function Equal-Path([string]$First, [string]$Second) {
    [System.IO.Path]::GetFullPath($First) -eq [System.IO.Path]::GetFullPath($Second)
}
function Group-Counts($Items, [string]$Property) {
    $taskMap = [ordered]@{}
    foreach ($taskGroup in @($Items | Group-Object -Property $Property | Sort-Object Name)) {
        $taskMap[$taskGroup.Name] = $taskGroup.Count
    }
    $taskMap
}
function Write-NewJson([string]$Name, $Value) {
    $taskPath = Join-Path $batchRoot $Name
    Assert-True (-not (Test-Path -LiteralPath $taskPath)) "Existing file: $taskPath"
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $taskPath -Encoding utf8
}
Add-Type -AssemblyName System.Drawing
$records = @(Read-Json (Join-Path $batchRoot '전체호출_v3.json'))
$plan = @(Read-Json (Join-Path $batchRoot '실행계획_v3.json'))
$relocation = Read-Json (Join-Path $batchRoot '결과선정_v3.json')
$sourceCheckpoint = @(Read-Json (Join-Path $batchRoot '원본검산_중간_v3.json'))
$lineage = @(Import-Csv -LiteralPath $sourceList -Encoding UTF8)
Assert-True ($records.Count -eq 201) 'Call count'
Assert-True ($plan.Count -eq 150 -and $lineage.Count -eq 150 -and $sourceCheckpoint.Count -eq 150) 'Source/plan count'
Assert-True (@(Get-ChildItem -LiteralPath $sourceRoot -File -Filter 'src3_*.jpg').Count -eq 150) 'Source file count'
Assert-True (@($records | Group-Object id,stage).Count -eq 201) 'Duplicate call key'
Assert-True (@($records | Group-Object generated_path).Count -eq 201) 'Duplicate raw path'
Assert-True (@($records | Group-Object sha256).Count -eq 201) 'Duplicate output hash'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '기록') -File -Filter '*.json').Count -eq 201) 'Record file count'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '사전프롬프트') -File -Filter '*.json').Count -eq 201) 'Prompt file count'
$sourceMeta = @{}
foreach ($taskRow in $lineage) {
    $taskNumber = [int]$taskRow.원본번호
    $taskId = 'src3_{0:d3}' -f $taskNumber
    Assert-True ($taskNumber -ge 1 -and $taskNumber -le 150) 'Source number'
    Assert-True ($taskRow.파일명 -ceq ($taskId+'.jpg')) 'Source filename'
    Assert-True (-not $sourceMeta.ContainsKey($taskId)) 'Duplicate source'
    Assert-True (-not [string]::IsNullOrWhiteSpace($taskRow.농장코드)) 'Farm empty'
    Assert-True (-not [string]::IsNullOrWhiteSpace($taskRow.AI허브_원천)) 'Lineage empty'
    $taskPath = Join-Path $sourceRoot $taskRow.파일명
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash
    $taskCheck = @($sourceCheckpoint | Where-Object source -CEQ $taskRow.파일명)
    Assert-True ($taskCheck.Count -eq 1 -and $taskCheck[0].sha256 -eq $taskHash) 'Source changed since checkpoint'
    Assert-True ($taskCheck[0].bytes -eq (Get-Item -LiteralPath $taskPath).Length) 'Source bytes changed'
    $taskDims = Read-Dimensions $taskPath
    $sourceMeta[$taskId] = [pscustomobject]@{
        source=$taskRow.파일명; sha256=$taskHash; width=$taskDims.width; height=$taskDims.height
        farm=$taskRow.농장코드; aihub_source=$taskRow.AI허브_원천
    }
}
Assert-True ($sourceMeta.Count -eq 150) 'Source coverage'
Assert-True (@($sourceMeta.Values | Group-Object sha256).Count -eq 150) 'Duplicate source hash'
$selected = @()
$middle = @($records | Where-Object stage -CEQ 'stage1' | Sort-Object n)
Assert-True ($middle.Count -eq 50) 'Intermediate count'
for ($taskNumber = 1; $taskNumber -le 150; $taskNumber++) {
    $taskId = 'src3_{0:d3}' -f $taskNumber
    $taskStage = if ($taskNumber -eq 127) { 'final_retry1' } else { 'final' }
    $taskChosen = @($records | Where-Object { $_.id -ceq $taskId -and $_.stage -ceq $taskStage })
    Assert-True ($taskChosen.Count -eq 1) "Final selection $taskId"
    $taskRecord = $taskChosen[0]
    $taskMethod = if ($taskNumber -le 50) { 'A' } elseif ($taskNumber -le 100) { 'B' } else { 'C' }
    $taskType = if ($taskNumber -le 100) { '3_부분생성' } else { '4_부분스타일' }
    Assert-True ($taskRecord.n -eq $taskNumber -and $taskRecord.method -ceq $taskMethod -and $taskRecord.type -ceq $taskType) 'Method/range/type'
    Assert-True ($taskRecord.source -ceq ($taskId+'.jpg')) 'Final source'
    $selected += $taskRecord
}
Assert-True (@($selected | Group-Object source).Count -eq 150) 'Final one per source'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '결과_GPT\3_부분생성') -File -Filter '*.png').Count -eq 100) 'Partial generation count'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '결과_GPT\4_부분스타일') -File -Filter '*.png').Count -eq 50) 'Partial style count'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '중간_GPT') -File -Filter '*.png').Count -eq 50) 'Middle folder count'
Assert-True (@(Get-ChildItem -LiteralPath (Join-Path $batchRoot '보류_GPT') -File -Filter '*.png').Count -eq 1) 'Held folder count'
Assert-True (-not (Test-Path -LiteralPath $relocation.original_result)) 'Original relocated copy still in finals'
$fileMeta = @()
$callMeta = @{}
$taskRootFull = [System.IO.Path]::GetFullPath($batchRoot).TrimEnd('\')+'\'
foreach ($taskRecord in $records) {
    $taskKey = $taskRecord.id+'_'+$taskRecord.stage
    $taskSaved = Read-Json (Join-Path $batchRoot ('기록\'+$taskKey+'.json'))
    $taskBefore = Read-Json (Join-Path $batchRoot ('사전프롬프트\'+$taskKey+'.json'))
    foreach ($taskField in @('n','id','source','method','type','action','size','stage','prompt','input','relative')) {
        Assert-True ($taskRecord.$taskField -ceq $taskSaved.$taskField) "Aggregate/record mismatch $taskKey $taskField"
        Assert-True ($taskRecord.$taskField -ceq $taskBefore.$taskField) "Preprompt/record mismatch $taskKey $taskField"
    }
    Assert-True ($taskRecord.tool -ceq 'GPT_builtin_imagegen') 'Tool provenance'
    Assert-True ($taskRecord.prompt.Contains($requiredPhrase)) "Required phrase $taskKey"
    $taskSource = $sourceMeta[$taskRecord.id]
    Assert-True ($null -ne $taskSource) 'Unknown source'
    if ($taskRecord.stage -ceq 'stage1') {
        Assert-True ($taskRecord.n -le 50 -and $taskRecord.method -ceq 'A') 'Unexpected stage1'
        Assert-True (Equal-Path $taskRecord.input (Join-Path $sourceRoot $taskRecord.source)) 'Stage1 input'
    } elseif ($taskRecord.method -ceq 'A') {
        $taskIntermediate = @($middle | Where-Object id -CEQ $taskRecord.id)
        Assert-True ($taskIntermediate.Count -eq 1) 'Missing intermediate'
        Assert-True (Equal-Path $taskRecord.input $taskIntermediate[0].result) 'A stage2 chaining'
        Assert-True ((Get-FileHash -LiteralPath $taskRecord.input -Algorithm SHA256).Hash -eq $taskIntermediate[0].sha256) 'A stage2 input hash'
    } else {
        Assert-True (Equal-Path $taskRecord.input (Join-Path $sourceRoot $taskRecord.source)) 'B/C input'
    }
    $taskActual = $taskRecord.result
    $taskRelative = $taskRecord.relative
    $taskRole = if ($taskRecord.stage -ceq 'stage1') { 'intermediate' } else { 'selected_final' }
    if ($taskRecord.id -ceq $relocation.id -and $taskRecord.stage -ceq $relocation.stage) {
        Assert-True (Equal-Path $taskRecord.result $relocation.original_result) 'Relocation source mismatch'
        Assert-True ($taskRecord.sha256 -eq $relocation.sha256) 'Relocation hash mismatch'
        $taskActual = $relocation.relocated_result
        $taskRelative = $relocation.relative
        $taskRole = 'held'
    }
    Assert-True ([System.IO.Path]::GetFullPath($taskActual).StartsWith($taskRootFull,[System.StringComparison]::OrdinalIgnoreCase)) 'Path outside batch'
    Assert-True (Equal-Path $taskActual (Join-Path $batchRoot $taskRelative)) 'Relative/absolute path mismatch'
    $taskHash = (Get-FileHash -LiteralPath $taskActual -Algorithm SHA256).Hash
    $taskRawHash = (Get-FileHash -LiteralPath $taskRecord.generated_path -Algorithm SHA256).Hash
    Assert-True ($taskHash -eq $taskRawHash -and $taskHash -eq $taskRecord.sha256) "Raw/copy/hash $taskKey"
    $taskBytes = (Get-Item -LiteralPath $taskActual).Length
    Assert-True ($taskBytes -eq $taskRecord.bytes) 'Output bytes'
    $taskDims = Read-Dimensions $taskActual -Png
    $taskAspectDelta = 100.0*[math]::Abs(($taskDims.width/[double]$taskDims.height)/($taskSource.width/[double]$taskSource.height)-1)
    $taskMeta = [pscustomobject]@{
        id=$taskRecord.id; stage=$taskRecord.stage; role=$taskRole; method=$taskRecord.method
        type=$taskRecord.type; action=$taskRecord.action; requested_size=$taskRecord.size
        style=$taskRecord.style; requested_intensity=$taskRecord.intensity; file=$taskRelative
        result=$taskActual; historical_result=$taskRecord.result; generated_path=$taskRecord.generated_path
        width=$taskDims.width; height=$taskDims.height; bytes=$taskBytes; sha256=$taskHash
        source=$taskSource.source; source_sha256=$taskSource.sha256; source_width=$taskSource.width; source_height=$taskSource.height
        farm=$taskSource.farm; aihub_source=$taskSource.aihub_source; raw_copy_identical=$true
        aspect_ratio_relative_deviation_percent=$taskAspectDelta; prompt=$taskRecord.prompt; input=$taskRecord.input
    }
    $fileMeta += $taskMeta
    $callMeta[$taskKey] = $taskMeta
}
$partial = @($selected | Where-Object method -NE 'C')
$stylePart = @($selected | Where-Object method -CEQ 'C')
$drawings = @($stylePart | Where-Object action -CEQ '스타일')
$colorOnly = @($stylePart | Where-Object action -CEQ '색만')
$actionCounts = Group-Counts $partial 'action'
Assert-True ($actionCounts['추가'] -eq 27 -and $actionCounts['지우기'] -eq 27 -and $actionCounts['바꾸기'] -eq 26 -and $actionCounts['갈변'] -eq 20) 'Action allocation'
Assert-True (@($partial | Where-Object size -CEQ '포기 하나 이상').Count -eq 60) 'Large requested count'
Assert-True ($drawings.Count -eq 40 -and $colorOnly.Count -eq 10) 'C drawing/color count'
Assert-True (@($stylePart | Where-Object intensity -CEQ '약하게').Count -eq 25 -and @($stylePart | Where-Object intensity -CEQ '뚜렷하게').Count -eq 25) 'Intensity allocation'
Assert-True (@($drawings | Where-Object intensity -CEQ '약하게').Count -eq 20 -and @($colorOnly | Where-Object intensity -CEQ '약하게').Count -eq 5) 'Intensity suballocation'
$styleGroups = @($drawings | Group-Object style)
Assert-True ($styleGroups.Count -eq 5 -and @($styleGroups | Where-Object Count -NE 8).Count -eq 0) 'Five styles equal'
$manifest = @()
$editRecords = @()
foreach ($taskRecord in $selected) {
    $taskMeta = $callMeta[$taskRecord.id+'_'+$taskRecord.stage]
    $taskPrompt = $taskRecord.prompt
    if ($taskRecord.method -ceq 'A') {
        $taskStage1 = @($middle | Where-Object id -CEQ $taskRecord.id)[0]
        $taskPrompt = "[A 1단계 실제 프롬프트]"+[Environment]::NewLine+$taskStage1.prompt+[Environment]::NewLine+"[A 2단계 실제 프롬프트]"+[Environment]::NewLine+$taskRecord.prompt
    }
    $manifest += [pscustomobject][ordered]@{
        id=$taskMeta.id; type=$taskMeta.type; file=$taskMeta.file; width=$taskMeta.width; height=$taskMeta.height
        bytes=$taskMeta.bytes; sha256=$taskMeta.sha256; source=$taskMeta.source; source_sha256=$taskMeta.source_sha256
        farm=$taskMeta.farm; aihub_source=$taskMeta.aihub_source; raw_copy_identical=$true; prompt=$taskPrompt; 방식=$taskMeta.method
    }
    $editRecords += [pscustomobject][ordered]@{
        도구='GPT'; 유형=$taskMeta.type; 결과파일=$taskMeta.file; 원본파일=$taskMeta.source; 프롬프트=$taskPrompt; 방식=$taskMeta.method
    }
}
$manifestPath = Join-Path $batchRoot 'manifest_v3.csv'
$recordsPath = Join-Path $batchRoot '기록_v3.csv'
if ($WriteCsv) {
    Assert-True (-not (Test-Path -LiteralPath $manifestPath) -and -not (Test-Path -LiteralPath $recordsPath)) 'CSV already exists'
    $manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8BOM
    $editRecords | Export-Csv -LiteralPath $recordsPath -NoTypeInformation -Encoding utf8BOM
}
$manifestRead = @(Import-Csv -LiteralPath $manifestPath -Encoding UTF8)
$recordsRead = @(Import-Csv -LiteralPath $recordsPath -Encoding UTF8)
Assert-True ($manifestRead.Count -eq 150 -and $recordsRead.Count -eq 150) 'CSV row counts'
$taskV2Manifest = @(Import-Csv -LiteralPath (Join-Path $v2Root 'manifest_v2.csv') -Encoding UTF8)
$taskV2Records = @(Import-Csv -LiteralPath (Join-Path $v2Root '기록_v2.csv') -Encoding UTF8)
$taskManifestHeaders = @($taskV2Manifest[0].PSObject.Properties.Name)+@('방식')
$taskRecordHeaders = @($taskV2Records[0].PSObject.Properties.Name)+@('방식')
Assert-True ((@($manifestRead[0].PSObject.Properties.Name)-join '|') -ceq ($taskManifestHeaders-join '|')) 'Manifest header v2+method'
Assert-True ((@($recordsRead[0].PSObject.Properties.Name)-join '|') -ceq ($taskRecordHeaders-join '|')) 'Record header v2+method'
for ($taskIndex=0; $taskIndex -lt 150; $taskIndex++) {
    foreach ($taskHeader in $taskManifestHeaders) {
        Assert-True ([string]$manifestRead[$taskIndex].$taskHeader -ceq [string]$manifest[$taskIndex].$taskHeader) "Manifest roundtrip $taskIndex $taskHeader"
    }
    foreach ($taskHeader in $taskRecordHeaders) {
        Assert-True ([string]$recordsRead[$taskIndex].$taskHeader -ceq [string]$editRecords[$taskIndex].$taskHeader) "Record roundtrip $taskIndex $taskHeader"
    }
}
$selectedMeta = @($fileMeta | Where-Object role -CEQ 'selected_final')
$summary = [ordered]@{
    status='PASS'; verified_at=(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'); image_tool='GPT_builtin_imagegen'
    actual_calls=$records.Count; selected_final_count=$selectedMeta.Count; intermediate_count=$middle.Count; held_count=1; retry_count=1
    tool_error_count=0; refusal_count=0; visually_unselected_count=1
    selected_by_method=(Group-Counts $selected 'method'); selected_by_type=(Group-Counts $selected 'type')
    partial_actions=$actionCounts; requested_partial_sizes=(Group-Counts $partial 'size')
    C_actions=(Group-Counts $stylePart 'action'); C_drawing_styles=(Group-Counts $drawings 'style')
    C_requested_intensities=(Group-Counts $stylePart 'intensity')
    source_count=150; source_hash_unique=150; final_source_unique=150
    farm_count=@($sourceMeta.Values | Group-Object farm).Count; farms=(Group-Counts $sourceMeta.Values 'farm')
    all_raw_copy_hash_matches=201; all_png_readable=201; all_output_hash_unique=201
    source_checkpoint_matches=150; A_stage2_intermediate_links=50; required_phrase_matches=201; preprompt_record_matches=201
    manifest_rows=150; records_rows=150; manifest_headers=$taskManifestHeaders; records_headers=$taskRecordHeaders
    selected_bytes=($selectedMeta | Measure-Object bytes -Sum).Sum; intermediate_bytes=(@($fileMeta | Where-Object role -CEQ 'intermediate') | Measure-Object bytes -Sum).Sum
    all_copy_bytes=($fileMeta | Measure-Object bytes -Sum).Sum
    final_width_min=($selectedMeta | Measure-Object width -Minimum).Minimum; final_width_max=($selectedMeta | Measure-Object width -Maximum).Maximum
    final_height_min=($selectedMeta | Measure-Object height -Minimum).Minimum; final_height_max=($selectedMeta | Measure-Object height -Maximum).Maximum
    max_final_aspect_relative_deviation_percent=($selectedMeta | Measure-Object aspect_ratio_relative_deviation_percent -Maximum).Maximum
    max_all_aspect_relative_deviation_percent=($fileMeta | Measure-Object aspect_ratio_relative_deviation_percent -Maximum).Maximum
    resize_crop_reencode_mask_composite_count=0; training_count=0; model_evaluation_count=0; submission_count=0
    d1_distribution_measured=$false; semantic_edit_sizes_measured=$false; OCR_measured=$false
    limitations=@('색/스타일 강도와 편집 크기는 요청값이며 의미적 전수 통과 보증이 아님.','중간 원본 해시 체크포인트는 A 완료 후 기록되어 실행 전 전체 원본 기준선이라고 하지 않음.','비율 차이는 생성 도구의 원출력이며 교정하지 않음.','d1·모델 성능·학습 효용은 측정하지 않음.')
}
if ($WriteCsv) {
    Write-NewJson '파일검산_v3.json' $fileMeta
    Write-NewJson '검증_v3.json' $summary
}
$summary | ConvertTo-Json -Depth 10

