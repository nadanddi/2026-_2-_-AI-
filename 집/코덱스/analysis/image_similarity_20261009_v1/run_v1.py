import sys, io, json, csv, hashlib, collections, argparse
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'클로드'/'research'))
# Repository bootstrap is resolved explicitly from own script location.
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from scipy.fft import dctn
HERE=Path(__file__).resolve().parent
LOCAL=ROOT/'집'/'코덱스'/'local'/'image_similarity_20261009_v1'
ZIP=Path('G:/내 드라이브/농업 AI 경진대회/이미지 미션/New_Sample.zip의 사본')
COMP=ROOT/'공용'/'대회자료'/'이미지데이터'/'참가자_배포'/'공개샘플'
LOCAL.mkdir(parents=True,exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
def save_json(name,obj):
 with (HERE/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def inspect(raw,name,meta=None):
 with Image.open(io.BytesIO(raw)) as im: im.verify()
 with Image.open(io.BytesIO(raw)) as im:
  mode=im.mode; size=im.size; fmt=im.format; orientation=im.getexif().get(274)
  rgb=ImageOps.exif_transpose(im).convert('RGB'); rgb.load()
  gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
  dc=dctn(gray,type=2,norm='ortho')[:8,:8].flatten()
  bits=dc>np.median(dc[1:]); bits[0]=False
  small=np.asarray(rgb.resize((128,128),Image.Resampling.BILINEAR))
  hist=np.concatenate([np.histogram(small[:,:,k],bins=8,range=(0,256))[0] for k in range(3)]).astype(float)
  hist/=hist.sum()
  row={'name':name,'width':size[0],'height':size[1],'display_width':rgb.width,'display_height':rgb.height,'mode':mode,'format':fmt,'orientation':orientation,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'pixel_sha256':hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),'phash_bits':bits.astype(int).tolist(),'histogram':hist.tolist(),'meta':meta}
  thumb=rgb.copy(); thumb.thumbnail((240,170))
  return row,thumb

def sheet(rows,thumbs,name,cols=6):
 cellw,cellh=250,205; nrows=(len(rows)+cols-1)//cols
 canvas=Image.new('RGB',(cols*cellw,nrows*cellh),'white'); draw=ImageDraw.Draw(canvas)
 for k,row in enumerate(rows):
  x=(k%cols)*cellw; y=(k//cols)*cellh
  th=thumbs[row['name']]; canvas.paste(th,(x+(240-th.width)//2,y))
  draw.text((x+4,y+173),row.get('short',row['name'].split('/')[-1])[:29],fill='black',font=FONT)
  draw.text((x+4,y+190),f"{row['width']}x{row['height']} {row.get('type','')}",fill='black',font=FONT)
 canvas.save(LOCAL/name)

def profile():
 ai=[]; co=[]; thumbs={}; metadata={}; errors=[]
 with ZipFile(ZIP) as z:
  infos=z.infolist(); save_json('zip_manifest_v1.json',[{'name':i.filename,'bytes':i.file_size,'crc':i.CRC} for i in infos if not i.is_dir()])
  for n in sorted(z.namelist()):
   if n.lower().endswith('.json'):
    m=json.loads(z.read(n).decode('utf-8-sig')); key=Path(n).stem
    if key in metadata: raise ValueError('duplicate JSON basename')
    metadata[key]=m
  for n in sorted(z.namelist()):
   if n.lower().endswith('.jpg'):
    try:
     key=Path(n).stem
     m=metadata.get(key)
     row,th=inspect(z.read(n),n,m['images'] if m else None)
     ai.append(row); thumbs[n]=th
    except Exception as e: errors.append({'name':n,'error':str(e)})
 labels=list(csv.DictReader((COMP/'sample_labels.csv').open(encoding='utf-8-sig')))
 for lab in labels:
  row,th=inspect((COMP/'images'/lab['file']).read_bytes(),lab['file'])
  row.update(label=int(lab['label']),type=lab['type']); co.append(row); thumbs[row['name']]=th
 assert not errors,errors
 assert len(ai)==340 and len(co)==21 and len(metadata)==340
 assert {Path(r['name']).stem for r in ai}==set(metadata)
 summary={}
 for group,rows in [('aihub',ai),('competition',co)]:
  summary[group]={'n':len(rows),'dimensions':dict(collections.Counter(f"{r['width']}x{r['height']}" for r in rows)),'modes':dict(collections.Counter(r['mode'] for r in rows)),'orientations':dict(collections.Counter(str(r['orientation']) for r in rows)),'unique_bytes':len({r['sha256'] for r in rows}),'unique_pixels':len({r['pixel_sha256'] for r in rows})}
 summary['metadata']={k:dict(collections.Counter(str(r['meta'].get(k)) for r in ai)) for k in ['farm_id','crops','kind_type','growth_stage','leaf','plant_body']}
 dates=[r['meta'].get('date_captured') for r in ai]
 summary['metadata']['date_captured_min']=min(dates); summary['metadata']['date_captured_max']=max(dates)
 summary['cross_byte_duplicates']=sorted(set(r['sha256'] for r in ai)&set(r['sha256'] for r in co))
 summary['cross_pixel_duplicates']=sorted(set(r['pixel_sha256'] for r in ai)&set(r['pixel_sha256'] for r in co))
 summary['errors']=errors
 summary['cleaning']='none; all originals retained; EXIF orientation for visual descriptor only'
 save_json('profile_v1.json',summary); save_json('records_v1.json',{'aihub':ai,'competition':co})
 sheet(co,thumbs,'competition_all21_v1.jpg',cols=7)
 for page in range(8):
  part=ai[page*48:(page+1)*48]
  if part: sheet(part,thumbs,f'aihub_all_page{page+1}_v1.jpg')
 print(json.dumps(summary,ensure_ascii=False),flush=True)

def compare():
 data=json.loads((HERE/'records_v1.json').read_text(encoding='utf-8')); ai=data['aihub']; co=data['competition']
 a=np.array([r['phash_bits'] for r in ai],dtype=np.uint8); b=np.array([r['phash_bits'] for r in co],dtype=np.uint8)
 ham=np.count_nonzero(a[None,:,:]!=b[:,None,:],axis=2)
 ha=np.sqrt(np.array([r['histogram'] for r in ai])); hb=np.sqrt(np.array([r['histogram'] for r in co]))
 hell=np.sqrt(np.maximum(0,np.sum((hb[:,None,:]-ha[None,:,:])**2,axis=2)/2))
 np.savez_compressed(HERE/'distances_v1.npz',hamming=ham,hellinger=hell)
 result=[]
 for i,r in enumerate(co):
  picks={}
  for key,dist in [('phash',ham),('color',hell)]:
   order=np.argsort(dist[i],kind='stable')[:3]
   picks[key]=[{'name':ai[j]['name'],'distance':float(dist[i,j])} for j in order]
  result.append({'file':r['name'],'label':r['label'],'type':r['type'],**picks})
 summary={'pairs':ham.size,'min_phash':int(ham.min()),'pairs_phash_le6':int((ham<=6).sum()),'per_image':result,'interpretation':'pHash and RGB histogram distances are low-level descriptors, not semantic similarity percentages'}
 save_json('comparison_v1.json',summary)
 # Visual side-by-side: each competition image, nearest pHash, nearest color.
 thumb={}
 with ZipFile(ZIP) as z:
  for r in result:
   for key in ['phash','color']:
    n=r[key][0]['name']
    if n not in thumb:
     with Image.open(io.BytesIO(z.read(n))) as im:
      t=ImageOps.exif_transpose(im).convert('RGB'); t.thumbnail((440,300)); thumb[n]=t.copy()
 for r in co:
  with Image.open(COMP/'images'/r['name']) as im:
   t=ImageOps.exif_transpose(im).convert('RGB'); t.thumbnail((440,300)); thumb[r['name']]=t.copy()
 for group,part in [('real',[r for r in result if r['label']==0]),('positive1',[r for r in result if r['label']==1][:6]),('positive2',[r for r in result if r['label']==1][6:12]),('positive3',[r for r in result if r['label']==1][12:])]:
  canvas=Image.new('RGB',(1350,len(part)*340),'white'); d=ImageDraw.Draw(canvas)
  for k,r in enumerate(part):
   selections=[(r['file'],'competition '+r['type']),(r['phash'][0]['name'],f"pHash {r['phash'][0]['distance']:.0f}/64"),(r['color'][0]['name'],f"Color Hellinger {r['color'][0]['distance']:.3f}")]
   for col,(n,title) in enumerate(selections):
    t=thumb[n]; x=col*450; y=k*340
    canvas.paste(t,(x+(440-t.width)//2,y)); d.text((x+4,y+302),title,fill='black',font=FONT); d.text((x+4,y+321),Path(n).name,fill='black',font=FONT)
  canvas.save(LOCAL/f'nearest_{group}_v1.jpg')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['profile','compare']); args=p.parse_args()
 {'profile':profile,'compare':compare}[args.stage]()
