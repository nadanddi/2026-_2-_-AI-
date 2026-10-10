import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,shutil,io
from PIL import Image
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_six_type_qa_20261010_v1'
def sizes(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
dirs=[ROOT/'집/코덱스/local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1']]
before=sum(sizes(d) for d in dirs);new_before=sizes(LOCAL);added=0;entries=[]
inputname=sys.argv[1];records=json.loads((RUN/inputname).read_text(encoding='utf-8'));prompts={r['id']:r for r in json.loads((RUN/'prompt_registry_v1.json').read_text(encoding='utf-8'))}
for r in records:
    path=Path(r['generated_path']).resolve();allowed=Path(r'C:\Users\aozks\.codex\generated_images').resolve()
    if not path.is_relative_to(allowed) or path.stat().st_size>32*1024**2:raise ValueError('generated file budget/path')
    before_stat=path.stat()
    with path.open('rb') as f:raw=f.read(32*1024**2+1)
    after_stat=path.stat()
    if len(raw)>32*1024**2 or len(raw)!=before_stat.st_size or (before_stat.st_size,before_stat.st_mtime_ns)!=(after_stat.st_size,after_stat.st_mtime_ns):raise ValueError('unstable generated snapshot')
    digest=hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>16_000_000:raise ValueError('pixel budget')
        im.load();dims=list(im.size);mode=im.mode;alpha_range=im.getchannel('A').getextrema() if 'A' in im.getbands() else None
    dest=LOCAL/(r['id']+'_asset_v1.png')
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    else:
        added+=len(raw)
        if before+added>16*1024**3 or new_before+added>1024**3 or shutil.disk_usage(ROOT).free-len(raw)<30*1024**3:raise ValueError('cache/free budget')
        with dest.open('xb') as f:f.write(raw)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    prompt=prompts[r['id']]
    reference_hashes=[{'path':ref,'sha256':hashlib.sha256(Path(ref).read_bytes()).hexdigest()} for ref in prompt.get('referenced_image_paths',[])]
    entries.append(dict(r,visual_parent='none_text_only' if not reference_hashes else 'reference_image',reference_hashes=reference_hashes,administrative_source_index=prompt['source'],path=str(dest),bytes=len(raw),sha256=digest,size=dims,mode=mode,alpha_range=alpha_range,prompt=prompts[r['id']],default_generated_source_preserved=True))
outfile='assets_'+('first' if 'first' in inputname else 'full')+'_v1.json'
with (RUN/outfile).open('x',encoding='utf-8') as f:json.dump(entries,f,ensure_ascii=False,indent=2)
print(json.dumps({'assets':len(entries),'new_bytes':added,'output':outfile},ensure_ascii=False))
