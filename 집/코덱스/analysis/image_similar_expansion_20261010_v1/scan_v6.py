import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import io,json,csv,hashlib,time,threading,os,shutil,tarfile,collections
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch,torchvision
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name;RAW=LOCAL/'검색용_원본'
RAW.mkdir(parents=True,exist_ok=True)
STAGE=globals().get('STAGE','pilot');LIMIT=globals().get('LIMIT',1)
PLAN=RUN/'source_plan_v1.json';plan=json.loads(PLAN.read_text(encoding='utf-8'))
INVENTORY=json.loads((RUN.parent/'image_download_complete_20261010_v1/inventory_v1.json').read_text(encoding='utf-8'))
paths={r['name']:r['path'] for r in INVENTORY['rows'] if r['kind']=='TS'}
allowed={'AIF005','AIF006','AIF007','AIF010','AIF012','AIF013'}
assert all(r['farm_id'] in allowed for a in plan['archives'] for r in a['samples'])
start=time.monotonic();pending=0;readbytes=0;requests=0;payloadbytes=0
prior=RUN/'scan_pilot_v1.json'
results=json.loads(prior.read_text(encoding='utf-8'))['results'] if STAGE=='expanded' and prior.exists() else []
if results:
    q=json.loads(prior.read_text(encoding='utf-8'));readbytes=q['read_requested_bytes'];payloadbytes=q['payload_bytes'];requests=q['read_requests']
receipt=RUN/'io_journal_expanded_v2.jsonl'
for previous in [RUN/'io_receipt_expanded_v1.json',RUN/'io_receipt_expanded_v1.tmp']:
    if previous.exists():
        saved=json.loads(previous.read_text(encoding='utf-8'));readbytes=max(readbytes,saved['requested_bytes']);requests=max(requests,saved['requests']);payloadbytes=max(payloadbytes,saved['payload_bytes'])
if not receipt.exists():
    readbytes+=13337+1536
else:
    for line in receipt.read_text(encoding='utf-8').splitlines():
        try:saved=json.loads(line)
        except ValueError:continue
        readbytes=max(readbytes,saved['requested_bytes']);requests=max(requests,saved['requests']);payloadbytes=max(payloadbytes,saved['payload_bytes'])
def ledger():
    with receipt.open('a',encoding='utf-8') as journal:
        journal.write('\n'+json.dumps(dict(requested_bytes=readbytes,requests=requests,payload_bytes=payloadbytes,pending=bool(pending)))+'\n');journal.flush();os.fsync(journal.fileno())
errors=[];done=[]
localstart=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file())
newbytes=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())
def guard(n=0):
    assert newbytes+n<2*1024**3 and localstart+newbytes+n<16*1024**3
    assert shutil.disk_usage(ROOT).free-n>30*1024**3
def watchdog():
    while True:
        time.sleep(1)
        if pending and time.monotonic()-pending>90:
            print('I/O pending >90s: stopping process with partial files preserved',flush=True);os._exit(124)
threading.Thread(target=watchdog,daemon=True).start()
class Counted:
    def __init__(self,p):
        global pending
        guard();pending=time.monotonic();ledger()
        try:self.f=open(p,'rb')
        finally:pending=0
    def read(self,n=-1):
        global pending,readbytes,requests
        assert n>=0;guard();assert readbytes+n<3*1024**3
        requests+=1;readbytes+=n;pending=time.monotonic();ledger()
        try:return self.f.read(n)
        finally:pending=0
    def seek(self,*a):
        global pending
        guard();pending=time.monotonic();ledger()
        try:return self.f.seek(*a)
        finally:pending=0
    def tell(self):return self.f.tell()
    def close(self):self.f.close()
torch.set_num_threads(2)
asset=json.loads((RUN.parent/'image_ingestion_20261010_v1/pretrained_asset_v1.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()==asset['sha256']
model=torchvision.models.resnet18(weights=None);model.load_state_dict(torch.load(asset['path'],map_location='cpu',weights_only=True));model.fc=torch.nn.Identity();model.eval().cuda()
cos=np.cos(np.pi*(np.arange(32)[None,:]+.5)*np.arange(8)[:,None]/32)
def desc(im):
    rgb=np.asarray(im.resize((128,128),Image.Resampling.BILINEAR),dtype=np.int16)
    mag=float(((rgb[:,:,0]-rgb[:,:,1]>20)&(rgb[:,:,2]-rgb[:,:,1]>20)).mean())
    h=np.concatenate([np.histogram(rgb[:,:,c],bins=8,range=(0,256))[0] for c in range(3)]).astype(float);h/=h.sum()
    a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float);co=(cos@a@cos.T).flatten();b=co>np.median(co[1:]);b[0]=False
    t=torch.from_numpy(np.array(im.resize((224,224),Image.Resampling.BILINEAR),copy=True)).permute(2,0,1).float()/255
    t=(t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return mag,h,b,t
def embed(ts):
    with torch.inference_mode():e=np.concatenate([model(torch.stack(ts[i:i+8]).cuda()).cpu().numpy() for i in range(0,len(ts),8)])
    return e/np.linalg.norm(e,axis=1,keepdims=True)
COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플'
pub=list(csv.DictReader((COMP/'sample_labels.csv').open(encoding='utf-8-sig')));pt=[];ph=[];pb=[]
for r in pub:
    raw=(COMP/'images'/r['file']).read_bytes();r['sha256']=hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:im=ImageOps.exif_transpose(im).convert('RGB');_,h,b,t=desc(im)
    pt.append(t);ph.append(h);pb.append(b)
pe=embed(pt);ph=np.array(ph);pb=np.array(pb)
bad=set(json.loads((RUN.parent/'image_source_mapping_20261010_v1/quarantine_v2.json').read_text(encoding='utf-8'))['image_ids'])
def checkpoint():
    obj=dict(stage=STAGE,plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),results=results,errors=errors,completed_plan_units=done,read_requested_bytes=readbytes,read_requests=requests,payload_bytes=payloadbytes,new_output_bytes=newbytes,provider_hydration_bytes='not measured',elapsed_seconds=time.monotonic()-start,public=pub)
    target=RUN/('scan_'+STAGE+'_v2.json');temp=RUN/('checkpoint_'+STAGE+'_v2.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');
    for retry in range(20):
        try:os.replace(temp,target);break
        except PermissionError:
            if retry==19:raise
            time.sleep(.1)
for unit,a in enumerate(plan['archives'][:LIMIT]):
    if STAGE=='expanded' and unit==0:done.append(unit);continue
    if time.monotonic()-start>720:errors.append(dict(unit=unit,reason='12min next-unit stop'));break
    print('archive',unit,a['farm'],a['archive'],flush=True)
    cf=None
    try:
        cf=Counted(paths[a['archive']]);wanted={Path(r['member']).stem:r for r in a['samples']};found={}
        with tarfile.open(fileobj=cf,mode='r:') as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith('.jpg') and Path(m.name).stem in wanted:
                    assert Path(m.name).stem not in found;found[Path(m.name).stem]=m
            assert set(found)==set(wanted)
            ts=[];hs=[];bs=[];batch=[]
            for stem,r in wanted.items():
                assert r['farm_id'] in allowed and r['image_id'] not in bad
                m=found[stem];assert 0<m.size<16*1024**2
                assert payloadbytes+m.size<2*1024**3;payloadbytes+=m.size
                cf.seek(m.offset_data);raw=cf.read(m.size);assert len(raw)==m.size
                with Image.open(io.BytesIO(raw)) as im:
                    assert im.width*im.height<32*1024**2
                    size=list(im.size);im=ImageOps.exif_transpose(im).convert('RGB');im.load();mag,h,b,t=desc(im)
                sha=hashlib.sha256(raw).hexdigest();p=RAW/(a['archive'].removesuffix('.tar')+'__'+Path(m.name).name)
                rr=dict(r,source_archive=a['archive'],source_path=paths[a['archive']],source_member=m.name,offset=m.offset_data,bytes=len(raw),sha256=sha,actual_size=size,dimension_match=size==[int(r['width']),int(r['height'])],magenta=mag,path=str(p),public_exact_byte=sha in {z['sha256'] for z in pub})
                if mag>.35:rr['eligible']=False;rr['reason']='dominant_magenta';results.append(rr);continue
                guard(len(raw))
                if not p.exists():
                    with p.open('xb') as f:f.write(raw)
                    newbytes+=len(raw)
                else:assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
                rr['eligible']=not rr['public_exact_byte'];ts.append(t);hs.append(h);bs.append(b);batch.append(rr)
            if batch:
                e=embed(ts);dc=1-pe@e.T;cl=np.sqrt(np.maximum(0,((np.sqrt(ph[:,None,:])-np.sqrt(np.array(hs)[None,:,:]))**2).sum(2)/2));hd=np.count_nonzero(pb[:,None,:]!=np.array(bs)[None,:,:],axis=2);sc=.5*dc+.3*cl+.2*hd/63
                for j,r in enumerate(batch):
                    r['scores_by_public']=[dict(file=z['file'],type=z['type'],score=float(sc[i,j]),phash_distance=int(hd[i,j])) for i,z in enumerate(pub)]
                    r['nearest_real']=min((z for z in r['scores_by_public'] if z['type']=='real'),key=lambda z:z['score'])
                    r['public_near_candidate']=int(hd[:,j].min())<=6 or r['prefix_candidate']=='C22_B02_001'
                    results.append(r)
        done.append(unit)
    except Exception as e:errors.append(dict(unit=unit,archive=a['archive'],error=repr(e)));print('error',repr(e),flush=True)
    finally:
        if cf:cf.close()
    checkpoint()
checkpoint()
print(json.dumps(dict(stage=STAGE,examined=len(results),eligible=sum(r.get('eligible',False) for r in results),errors=errors,read_requested_bytes=readbytes,payload_bytes=payloadbytes),ensure_ascii=False),flush=True)
