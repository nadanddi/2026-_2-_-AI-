"""C1: generate one registered G_path view of h3 training real originals only."""
import sys,csv,json,io,random,zlib,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent; B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
sys.path.insert(0,str(B/'data/image'))
from im03c_make_fakes_v3 import dist_cover
from im03_make_fakes import jpeg_bytes
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main(mode):
    if mode=='generate':
        real=[r for r in read(L/'im_set_h3/manifest.csv') if int(r['label'])==0]
        raw={(r['tar'],r['file']):r for r in read(L/'im_h_raw/manifest.csv') if r['role']=='htrain'}
        assert len(real)==len(raw)==1800 and {(r['tar'],r['source']) for r in real}==set(raw)
        eval_sources={Path(r['source'].replace('\\','/')).name for sn in ('im_set_realg','im_set_realn') for r in read(L/sn/'manifest.csv')}
        assert not ({r['source'] for r in real}&eval_sources)
        pool=json.loads((HERE/'gpt_size_pool_v1.json').read_text(encoding='utf-8')); longs=[max(r['width'],r['height']) for r in pool]
        D=HERE/'im_set_c1g'; (D/'img').mkdir(parents=True,exist_ok=False)
        with (D/'manifest.csv').open('x',encoding='utf-8',newline='') as fo:
            fields=['file','label','kind','source','group','farm_code','tar','target_long','raw_sha256','image_sha256']
            w=csv.DictWriter(fo,fieldnames=fields);w.writeheader()
            for i,r in enumerate(real):
                p=L/'im_h_raw'/r['tar']/r['source']; rng=random.Random(zlib.crc32(('C1G|'+r['source']).encode()))
                with Image.open(p) as im: im=im.convert('RGB')
                target=rng.choice(longs); s=target/max(im.size)
                im=im.resize((max(1,round(im.width*s)),max(1,round(im.height*s))),Image.LANCZOS)
                b=io.BytesIO();im.save(b,'PNG');im=Image.open(io.BytesIO(b.getvalue())).convert('RGB')
                payload=jpeg_bytes(dist_cover(im,rng)); name=f'c1g_{i:05d}.jpg'; (D/'img'/name).write_bytes(payload)
                w.writerow(dict(file=name,label=0,kind='real',source=r['source'],group=r['group'],farm_code=r['farm_code'],tar=r['tar'],target_long=target,
                    raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),image_sha256=hashlib.sha256(payload).hexdigest()))
                if (i+1)%300==0:fo.flush();print('G_PATH',i+1,flush=True)
        print('G_GENERATED',len(real),flush=True)
    elif mode=='features':
        import im43_h_feats as F
        F.L=HERE;F.OUT=HERE/'feats';F.OUT.mkdir(exist_ok=False)
        assert not (F.OUT/'im_set_c1g.npz').exists()
        m=F.I.load_model(str(F.PK/'model'));F.run('im_set_c1g',m)
        with np.load(F.OUT/'im_set_c1g.npz') as z:
            assert len(z['files'])==1800 and z['ok'].all()
        print('C1_FEATURES_DONE',flush=True)
    else:raise ValueError(mode)
if __name__=='__main__':main(sys.argv[1])
