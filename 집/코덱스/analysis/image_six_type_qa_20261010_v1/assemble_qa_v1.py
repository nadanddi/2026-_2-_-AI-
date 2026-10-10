import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,io,shutil,collections
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local/image_six_type_qa_20261010_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
dirs=[ROOT/'집/코덱스/local'/n for n in ['image_ingestion_20261010_v1','image_runtime_20261010_v1','image_download_complete_20261010_v1','image_source_mapping_20261010_v1','image_six_type_qa_20261010_v1']]
baseline=sum(size(p) for p in dirs);newbaseline=size(LOCAL);written=0
def write(p,b):
    global written
    if written+len(b)>128*1024**2 or baseline+written+len(b)>16*1024**3 or newbaseline+written+len(b)>1024**3 or shutil.disk_usage(ROOT).free-len(b)<30*1024**3:raise ValueError('output resource gate')
    with p.open('xb') as f:f.write(b)
    written+=len(b)
def encoded(im,format='PNG'):
    b=io.BytesIO();im.save(b,format=format,**({'quality':90,'subsampling':2} if format=='JPEG' else {}));return b.getvalue()
sources=json.loads((RUN/'prepared_sources_v1.json').read_text(encoding='utf-8'))
assets={r['id']:r for r in json.loads((RUN/'assets_full_v1.json').read_text(encoding='utf-8'))}
ledger=json.loads((RUN/'service_call_ledger_final_v1.json').read_text(encoding='utf-8'));assert ledger['attempted']==ledger['success']==8 and ledger['failure']==0
def asset(id,mode='RGB'):
    r=assets[id];assert sha(r['path'])==r['sha256']
    with Image.open(r['path']) as im:im.load();return im.convert(mode)
out=LOCAL/'qa';out.mkdir(exist_ok=False);records=[];tiles=[]
types=['real','fullgen','composite','filter','fullstyle','partialgen','partialstyle']
for i,s in enumerate(sources):
    assert sha(s['qa_source_path'])==s['qa_source_sha256'] and sha(s['mask_path'])==s['mask_sha256']
    with Image.open(s['qa_source_path']) as im:base=im.convert('RGB')
    a=np.asarray(base);x0,y0,x1,y1=s['content_bbox'];content=np.zeros((512,512),bool);content[y0:y1,x0:x1]=True
    with Image.open(s['mask_path']) as im:mask=np.asarray(im.convert('L'))>0
    style=asset(f'style_{i:02d}').resize((512,512),Image.Resampling.LANCZOS)
    partial=asset(f'partialgen_{i:02d}').resize((512,512),Image.Resampling.LANCZOS)
    gen=asset(f'fullgen_{i:02d}');full=Image.new('RGB',(512,512));full.paste(gen.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),(x0,y0))
    rgba=asset(f'cutout_{i:02d}','RGBA');alpha=np.asarray(rgba.getchannel('A'))
    if not (alpha.min()==0 and alpha.max()>0 and np.count_nonzero(alpha)>0):raise ValueError('cutout not transparent/empty')
    donor=rgba.crop(rgba.getbbox());donor.thumbnail((192,192),Image.Resampling.LANCZOS)
    composed=base.convert('RGBA');composed.alpha_composite(donor,(160+(192-donor.width)//2,160+(192-donor.height)//2));composed=composed.convert('RGB')
    pg=a.copy();pg[mask]=np.asarray(partial)[mask];ps=a.copy();ps[mask]=np.asarray(style)[mask]
    variants=[base,full,composed,base.filter(ImageFilter.GaussianBlur(1.0)),style,Image.fromarray(pg),Image.fromarray(ps)]
    asset_ids=[[],[f'fullgen_{i:02d}'],[f'cutout_{i:02d}'],[],[f'style_{i:02d}'],[f'partialgen_{i:02d}'],[f'style_{i:02d}']]
    real_decoded=None
    for j,(kind,im) in enumerate(zip(types,variants)):
        arr=np.asarray(im).copy();arr[~content]=a[~content];im=Image.fromarray(arr)
        changed=np.any(arr!=a,axis=2);is_partial=kind in ['partialgen','partialstyle','composite'];outside=int(np.count_nonzero(changed & ~mask)) if is_partial else None
        if is_partial and outside!=0:raise ValueError('pre-encode mask leak')
        pre=out/f'qa_{i:02d}_{j:02d}_pre_v1.png';write(pre,encoded(im))
        path=out/f'qa_{i:02d}_{j:02d}_v1.jpg';write(path,encoded(im,'JPEG'))
        with Image.open(path) as final:decoded=np.asarray(final.convert('RGB'));qtables=final.quantization
        if j==0:real_decoded=decoded
        final_changed=np.any(decoded!=real_decoded,axis=2);withheld=j>0 and not np.any(final_changed)
        expanded=np.zeros_like(mask);expanded[144:368,144:368]=True
        outside_final=int(np.count_nonzero(final_changed & ~mask)) if is_partial else None
        far_outside=int(np.count_nonzero(final_changed & ~expanded)) if is_partial else None
        if is_partial and far_outside!=0:raise ValueError('JPEG leak beyond conservative16px mask halo')
        farpad=np.ones_like(mask);farpad[max(0,y0-16):min(512,y1+16),max(0,x0-16):min(512,x1+16)]=False
        if np.any(final_changed & farpad):raise ValueError('far padding inconsistent')
        cm=decoded[content].astype('int16');magenta_fraction=float(np.mean((cm[:,0]-cm[:,1]>20)&(cm[:,2]-cm[:,1]>20)))
        record={'sample_id':f'qa_{i:02d}_{j:02d}','label':int(j>0),'positive_type':kind,'path':str(path),'sha256':sha(path),'pre_path':str(pre),'pre_sha256':sha(pre),'source_id':s['image_id'],'farm_id':s['farm_id'],'component':s['component'],'administrative_source_recipe':s['lighting_recipe'],'visual_parents':[] if kind=='fullgen' else [s['qa_source_sha256']],'source_sha256':s['qa_source_sha256'],'asset_ids':asset_ids[j],'asset_sha256':[assets[k]['sha256'] for k in asset_ids[j]],'source_individual_edit_history':'unverified','target_basis':'camera-source provenance assumption' if j==0 else 'recorded intentional production operation','mask_path':s['mask_path'] if is_partial else None,'mask_sha256':s['mask_sha256'] if is_partial else None,'mask_fraction':s['mask_fraction'] if is_partial else None,'changed_pixels_pre':int(changed.sum()),'changed_pixels_final':int(final_changed.sum()),'final_changed_fraction':float(final_changed.mean()),'outside_mask_changed_pre':outside,'outside_mask_changed_final_JPEG':outside_final,'outside_16px_mask_halo_changed_final':far_outside,'pad_pixels_identical_pre':True,'far_pad_changed_final':int(np.count_nonzero(final_changed & farpad)),'magenta_fraction_on_content':magenta_fraction,'common_export':{'input_size':[512,512],'JPEG_quality':90,'subsampling':2,'content_bbox':s['content_bbox'],'quantization':qtables},'withheld':withheld,'mask_controls':'pre-JPEG exact; decoded JPEG16px halo reported, not literal pixel-exact outside mask','training_role':'train_contract_only; no evaluation'}
        records.append(record);tiles.append((kind,s['image_id'],im))
assert len(records)==14 and not any(r['withheld'] for r in records)
font=ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf',16);canvas=Image.new('RGB',(7*260,2*292),'white');draw=ImageDraw.Draw(canvas)
names={'real':'실사진','fullgen':'전체 생성','composite':'합성','filter':'의도 필터','fullstyle':'전체 스타일','partialgen':'부분 생성','partialstyle':'부분 스타일'}
for k,(kind,id,im) in enumerate(tiles):
    thumb=im.resize((250,250));x=(k%7)*260;y=(k//7)*292;canvas.paste(thumb,(x,y));draw.text((x+3,y+253),names[kind]+' · '+id,font=font,fill='black')
contact=LOCAL/'six_type_contact_v1.jpg';write(contact,encoded(canvas,'JPEG'))
manifest={'rows':records,'type_counts':dict(collections.Counter(r['positive_type'] for r in records)),'label_counts':dict(collections.Counter(r['label'] for r in records)),'withheld':sum(r['withheld'] for r in records),'services':8,'source_count':2,'scope':'14 production QA files from2sources, no validation performance','contact':str(contact),'new_output_bytes':written,'B0_threshold_ge_05_training_distribution_accuracy':12/14}
with (RUN/'qa_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in manifest.items() if k!='rows'},ensure_ascii=False))
