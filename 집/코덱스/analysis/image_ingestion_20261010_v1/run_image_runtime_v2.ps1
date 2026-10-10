param([Parameter(Mandatory=$true)][string]$Script)
$ErrorActionPreference='Stop'
$repositoryRoot='C:\work\farmai'
$ownRoot=[IO.Path]::GetFullPath((Join-Path $repositoryRoot '집\코덱스'))
$scriptPath=[IO.Path]::GetFullPath($Script)
if (-not $scriptPath.StartsWith($ownRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Script must be under own Codex directory' }
$runtimeRoot=Join-Path $ownRoot 'local\image_runtime_20261010_v1'
$env:PYTHONPATH=''
$env:TORCH_HOME=Join-Path $runtimeRoot 'torch_assets'
$env:TEMP=Join-Path $runtimeRoot 'tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $runtimeRoot 'pip_cache'
& (Join-Path $runtimeRoot 'venv\Scripts\python.exe') -I -X utf8 -c 'import numpy,PIL.Image,torch,torchvision,runpy,sys;runpy.run_path(sys.argv[1],run_name=sys.argv[2])' $scriptPath __main__
exit $LASTEXITCODE

