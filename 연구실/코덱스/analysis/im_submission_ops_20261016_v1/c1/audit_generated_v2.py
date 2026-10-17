"""Mechanical G_path provenance audit, no classifier outcomes."""
import csv,json,hashlib,io
from pathlib import Path
from PIL import Image,JpegImagePlugin
HERE=Path(__file__).resolve().parent;L=Path.home()/'storage/nadan/farmai_img/data/local'
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
rows=read(HERE/'im_set_c1g/manifest.csv');original=[r for r in read(L/'im_set_h3/manifest.csv') if int(r['label'])==0]
assert len(rows)==len(original)==1800 and len({r['file'] for r in rows})==1800
pool=HERE/'gpt_size_pool_v1.json';assert hashlib.sha256(pool.read_bytes()).hexdigest()=='15a32cfc8997eba634495693ba11a770d2073a710a9e76aa115ff9af435066d0'
longs={max(r['width'],r['height']) for r in json.loads(pool.read_text(encoding='utf-8'))}
b=io.BytesIO();Image.new('RGB',(64,64)).save(b,'JPEG',quality=92);q=Image.open(io.BytesIO(b.getvalue())).quantization
for r,o in zip(rows,original):
    assert int(r['label'])==0 and r['kind']=='real'
    assert all(r[k]==o[k] for k in ('source','group','tar','farm_code'))
    p=HERE/'im_set_c1g/img'/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']
    assert int(r['target_long']) in longs and len(r['raw_sha256'])==64
    with Image.open(p) as im:assert im.size in ((768,1024),(1024,768)) and im.format=='JPEG' and im.quantization==q and JpegImagePlugin.get_sampling(im)==2
result=dict(pass_all=True,n=len(rows),all_real_training_sources=True,source_group_identical=True,image_sha256_verified=1800,gate_valid=1800,pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((HERE/'im_set_c1g/manifest.csv').read_bytes()).hexdigest())
with (HERE/'generated_audit_v2.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
