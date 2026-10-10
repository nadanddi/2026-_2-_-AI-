import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,shutil
from PIL import Image
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
stage='full' if (RUN/'service_outputs_full_v1.json').exists() else 'first'
rows=json.loads((RUN/f'service_outputs_{stage}_v1.json').read_text(encoding='utf-8'));receipt=[]
for row in rows:
    p=Path(row['generated_path']);before=p.stat()
    with p.open('rb') as f:raw=f.read(32*1024**2+1)
    assert len(raw)<=32*1024**2 and len(raw)==before.st_size==p.stat().st_size
    with Image.open(io.BytesIO(raw)) as im:assert im.width*im.height<=16*1024**2;im.load();dims=list(im.size)
    digest=hashlib.sha256(raw).hexdigest();dest=LOCAL/(row['id']+'_asset_v1.png')
    size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    total=sum(f.stat().st_size for name in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1',RUN.name] for f in (ROOT/'집/코덱스/local'/name).rglob('*') if f.is_file())
    if not dest.exists():
        assert size+len(raw)<256*1024**2 and total+len(raw)<16*1024**3 and shutil.disk_usage(ROOT).free-len(raw)>30*1024**3
        with dest.open('xb') as f:f.write(raw)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    receipt.append(dict(row,path=str(dest),sha256=digest,bytes=len(raw),size=dims,reference_sha256=hashlib.sha256(Path(row['reference']).read_bytes()).hexdigest() if row['reference'] else None,visual_parent='selected AIhub photo' if row['reference'] else 'none; text-only generation',composite_scope='requested foreground preservation/background swap; segmentation alpha not exposed' if row['type']=='composite' else None))
with (RUN/f'assets_{stage}_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps({'stage':stage,'files':len(receipt),'default_generated_bytes':sum(r['bytes'] for r in receipt)}))
