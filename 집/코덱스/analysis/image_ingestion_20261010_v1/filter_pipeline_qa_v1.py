import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import json,hashlib,io,collections
import numpy as np
from PIL import Image,ImageOps,ImageFilter,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집'/'코덱스'/'local'/'image_ingestion_20261010_v1'
source=json.loads((RUN/'source_qa_v1.json').read_text(encoding='utf-8'))
selected=source['selected'];outdir=LOCAL/'filter_pipeline_qa_v1';outdir.mkdir(exist_ok=False)
font=ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf',16)
report=[];sheet=Image.new('RGB',(4*320,2*290),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(selected):
    p=Path(r['local_path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
    with Image.open(io.BytesIO(raw)) as im:
        real=ImageOps.exif_transpose(im).convert('RGB');real.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    method=['intent_gaussian_blur','intent_gaussian_noise','intent_salt_pepper','intent_JPEG'][i%4]
    rng=np.random.default_rng(20261010+i);a=np.asarray(real,dtype=np.uint8)
    params={}
    if method=='intent_gaussian_blur':fake=real.filter(ImageFilter.GaussianBlur(1.0));params={'sigma_px':1.0}
    elif method=='intent_gaussian_noise':
        fake=Image.fromarray(np.clip(a.astype(float)+rng.normal(0,5,a.shape),0,255).round().astype('uint8'));params={'sd_0_255':5,'seed':20261010+i}
    elif method=='intent_salt_pepper':
        mask=rng.random(a.shape[:2]);b=a.copy();b[mask<.001]=0;b[(mask>=.001)&(mask<.002)]=255;fake=Image.fromarray(b);params={'probability':.002,'seed':20261010+i}
    else:
        buf=io.BytesIO();real.save(buf,format='JPEG',quality=60,subsampling=2);buf.seek(0)
        with Image.open(buf) as im:fake=im.convert('RGB')
        params={'inner_quality':60,'inner_subsampling':2}
    paths=[];decoded=[]
    for label,im in [(0,real),(1,fake)]:
        dest=outdir/f'qa_{i:02d}_{label}.jpg';im.save(dest,format='JPEG',quality=90,subsampling=2)
        with Image.open(dest) as x:decoded.append(np.asarray(x.convert('RGB')))
        paths.append({'label':label,'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    delta=decoded[1].astype('int16')-decoded[0].astype('int16');changed=int(np.count_nonzero(np.any(delta!=0,axis=2)))
    report.append({'source':r['fname'],'farm':r['farm_id'],'cultivar':r['kind_type'],'parent_sha256':r['sha256'],'prefix_candidate_unverified':r['prefix_candidate'],'label_provenance':'source original from user-provided AIHub raw archive vs intentional filter operation','positive_type':'filter','method':method,'params':params,'common_export':{'longside_limit':1024,'format':'JPEG','quality':90,'subsampling':2,'label_independent':True},'output':paths,'changed_pixels':changed,'pixel_count':delta.shape[0]*delta.shape[1],'mean_absolute_final_RGB_delta':float(np.abs(delta).mean()),'withheld':changed==0,'scope':'training pipeline QA only; not benchmark or six-type coverage'})
    thumb=real.copy();thumb.thumbnail((310,240));x=(i%4)*320;y=(i//4)*290;sheet.paste(thumb,(x,y));draw.text((x+4,y+240),r['kind_type']+' '+r['farm_id']+' '+r['date_captured'][:10],fill='black',font=font)
    draw.text((x+4,y+262),r['prefix_candidate'],fill='black',font=font)
assert len(report)==8
sheet.save(LOCAL/'train_real_QA_contact_v1.jpg',quality=90)
(RUN/'filter_pipeline_qa_v1.json').write_text(json.dumps({'pairs':report,'real':8,'filter_positive':8,'coverage':'filter subtype only; other five positive types not made','training':False,'evaluation':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pairs':len(report),'real':8,'positive_filter':8,'withheld':sum(r['withheld'] for r in report),'metadata_dimension_mismatch':sum(r['metadata_dimension_mismatch'] for r in selected),'original_sha_verified':8,'min_changed_pixels':min(r['changed_pixels'] for r in report)},ensure_ascii=False))
