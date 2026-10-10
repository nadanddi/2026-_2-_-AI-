import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import io,json,csv,hashlib,collections,shutil
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import torch,torchvision
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
LOCAL.mkdir(parents=True,exist_ok=True)
PREV=RUN.parent/'image_source_mapping_20261010_v1';CACHE=ROOT/'집/코덱스/local'/PREV.name
COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플'
def save(name,obj):
    with (RUN/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def guard(n):
    size=sum(f.stat().st_size for f in LOCAL.rglob('*') if f.is_file())
    total=sum(f.stat().st_size for name in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1',RUN.name] for f in (ROOT/'집/코덱스/local'/name).rglob('*') if f.is_file())
    assert size+n<256*1024**2 and total+n<16*1024**3 and shutil.disk_usage(ROOT).free-n>30*1024**3
def output(im,name,fmt):
    b=io.BytesIO();im.save(b,format=fmt,**({'quality':90,'subsampling':2} if fmt=='JPEG' else {}));raw=b.getvalue();guard(len(raw));p=LOCAL/name
    with p.open('xb') as f:f.write(raw)
    return str(p),hashlib.sha256(raw).hexdigest()
torch.set_num_threads(2)
asset=json.loads((RUN.parent/'image_ingestion_20261010_v1/pretrained_asset_v1.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()==asset['sha256']
model=torchvision.models.resnet18(weights=None);model.load_state_dict(torch.load(asset['path'],map_location='cpu',weights_only=True));model.fc=torch.nn.Identity();model.eval().cuda()
cos=np.cos(np.pi*(np.arange(32)[None,:]+.5)*np.arange(8)[:,None]/32)
def descriptors(im):
    rgb=np.asarray(im.resize((128,128),Image.Resampling.BILINEAR),dtype=np.int16)
    mag=float(((rgb[:,:,0]-rgb[:,:,1]>20)&(rgb[:,:,2]-rgb[:,:,1]>20)).mean())
    hist=np.concatenate([np.histogram(rgb[:,:,c],bins=8,range=(0,256))[0] for c in range(3)]).astype(float);hist/=hist.sum()
    grey=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    coeff=(cos@grey@cos.T).flatten();bits=(coeff>np.median(coeff[1:]));bits[0]=False
    a=np.array(im.resize((224,224),Image.Resampling.BILINEAR),copy=True)
    t=torch.from_numpy(a).permute(2,0,1).float()/255
    return mag,hist,bits,(t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
public=list(csv.DictReader((COMP/'sample_labels.csv').open(encoding='utf-8-sig')))
allrows=[];tensors=[];thumbs={};dcts=[];hists=[]
for r in public:
    p=COMP/'images'/r['file'];raw=p.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.load()
    mag,h,b,t=descriptors(im);r.update(sha256=hashlib.sha256(raw).hexdigest(),magenta=mag);allrows.append(r);tensors.append(t);dcts.append(b);hists.append(h)
    th=im.copy();th.thumbnail((170,200));thumbs[r['file']]=th
mapped=json.loads((PREV/'mapping_v1.json').read_text(encoding='utf-8'))['mapped']
bad=set(json.loads((PREV/'quarantine_v2.json').read_text(encoding='utf-8'))['image_ids'])
lineage={r['image_id']:r['component'] for r in json.loads((PREV/'source_lineage_v1.json').read_text(encoding='utf-8'))}
io_bytes=0;failures=[]
for n,r0 in enumerate(mapped):
    r=dict(r0);r['component']=lineage[r['image_id']]
    with (CACHE/r['source_archive']).open('rb') as f:f.seek(r['offset']);raw=f.read(r['bytes'])
    io_bytes+=len(raw);assert io_bytes<2*1024**3 and hashlib.sha256(raw).hexdigest()==r['payload_sha256']
    with Image.open(io.BytesIO(raw)) as im:
        assert im.width*im.height<=32*1024**2
        size=im.size;im=ImageOps.exif_transpose(im).convert('RGB');im.load()
    mag,h,b,t=descriptors(im);r.update(actual_size=list(size),dimension_match=size==(int(r['width']),int(r['height'])),magenta=mag)
    reasons=[]
    if r['image_id'] in bad:reasons.append('quarantine')
    if not r['dimension_match']:reasons.append('dimension_mismatch')
    if mag>.35:reasons.append('dominant_magenta')
    if r['prefix_candidate']=='C22_B02_001':reasons.append('prior_public_scene_family_suspect')
    r['excluded_reasons']=reasons
    allrows.append(r);tensors.append(t);dcts.append(b);hists.append(h)
    th=im.copy();th.thumbnail((170,200));thumbs[r['image_id']]=th
    if (n+1)%100==0:print('decoded',n+1,flush=True)
emb=[]
with torch.inference_mode():
    for i in range(0,len(tensors),8):
        e=model(torch.stack(tensors[i:i+8]).cuda()).cpu().numpy();emb.append(e)
features=np.concatenate(emb);features/=np.linalg.norm(features,axis=1,keepdims=True)
hists=np.array(hists);bits=np.array(dcts);pub=allrows[:21];candidates=allrows[21:]
distance=1-features[:21]@features[21:].T
colors=np.sqrt(np.maximum(0,((np.sqrt(hists[:21,None,:])-np.sqrt(hists[None,21:,:]))**2).sum(2)/2))
hashes=np.count_nonzero(bits[:21,None,:]!=bits[None,21:,:],axis=2)
scores=.5*distance+.3*colors+.2*hashes/63
for j,r in enumerate(candidates):
    near=int(np.argmin(hashes[:,j]));r['nearest_public_phash']={'file':pub[near]['file'],'distance':int(hashes[near,j])}
    if hashes[near,j]<=6:r['excluded_reasons'].append('near_public_phash_candidate')
    if r['payload_sha256'] in {p['sha256'] for p in pub}:r['excluded_reasons'].append('public_exact_byte')
    r['eligible']=not r['excluded_reasons']
rankings=[];chosen=[];prefixes=set()
for i,p in enumerate(pub):
    order=sorted([j for j,r in enumerate(candidates) if r['eligible']],key=lambda j:(float(scores[i,j]),candidates[j]['image_id']))
    top=[dict(candidates[j],score=float(scores[i,j]),embedding_cos_distance=float(distance[i,j]),color_Hellinger=float(colors[i,j]),phash_distance=int(hashes[i,j])) for j in order[:5]]
    rankings.append({'public_file':p['file'],'type':p['type'],'top5':top})
    if int(p['label'])==0:
        for row in top:
            if row['prefix_candidate'] not in prefixes:
                prefixes.add(row['prefix_candidate']);chosen.append(dict(row,selection_reference=p['file']));break
def contact(rows,name):
    canvas=Image.new('RGB',(6*200,((len(rows)+5)//6)*235),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13)
    for k,r in enumerate(rows):
        x=(k%6)*200;y=(k//6)*235;th=thumbs[r['image_id']];canvas.paste(th,(x+(200-th.width)//2,y));draw.text((x+3,y+201),r['image_id']+' '+r['prefix_candidate'],fill='black',font=font);draw.text((x+3,y+218),'분홍비율 %.3f'%r['magenta'],fill='black',font=font)
    output(canvas,name,'JPEG')
contact([r for q in rankings[:3] for r in q['top5']],'real_reference_candidates_v1.jpg')
contact(chosen,'chosen_three_candidates_v1.jpg')
save('selection_v1.json',{'search_scope':'cached VS gold2/seol3 mapped1000 only','public_observation_used':True,'public':pub,'candidates':candidates,'rankings':rankings,'chosen_three_candidates':chosen,'raw_payload_read_bytes':io_bytes,'eligible':sum(r['eligible'] for r in candidates),'exclusions':dict(collections.Counter(x for r in candidates for x in r['excluded_reasons'])),'weights_asset_sha256':asset['sha256'],'score':'0.5 cosine_distance +0.3 RGB_Hellinger +0.2 phash/63; no similarity percent','scope':'source/style selection, no fitted classifier/accuracy'})
print(json.dumps({'candidates':1000,'eligible':sum(r['eligible'] for r in candidates),'chosen':[(r['image_id'],r['selection_reference'],r['prefix_candidate']) for r in chosen]},ensure_ascii=False))
