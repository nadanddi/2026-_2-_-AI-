"""Read-only N9 reproduction and dataset metadata for A2."""
import sys,io,json
import numpy as np
from PIL import Image,JpegImagePlugin
from tokens_v1 import B,L,read
sys.path.insert(0,str(B/'data/image'))
sys.argv=[sys.argv[0]]
import im44_h_train_eval as H
buf=io.BytesIO();Image.new('RGB',(64,64)).save(buf,'JPEG',quality=92)
Q=Image.open(io.BytesIO(buf.getvalue())).quantization
def gate(p):
    with Image.open(p) as im:return im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==Q
def metadata(sn):
    if sn=='pub21':
        p=B.parent/'공용/대회자료/이미지데이터/참가자_배포/공개샘플';rr=read(p/'sample_labels.csv');return rr,[p/'images'/r['file'] for r in rr]
    if sn=='jig500':
        ps=sorted((L/'preds/_jig_hold').glob('j*.jpg'));return [dict(file=p.name,label='0',kind='jig',group='unknown') for p in ps],ps
    d=L/'sealed_c3'/sn[7:] if sn.startswith('sealed_') else L/sn
    rr=read(d/'manifest.csv');return rr,[d/'img'/r['file'] for r in rr]
def n9(F):
    M={}
    with np.load(L/'preds/h_model_h2.npz') as z:
        for k in z.files:a,c=k.split('__');M.setdefault(a,{})[c]=z[k]
    with np.load(L/'preds/ip_model_v1.npz') as z:gr={k.split('__')[1]:z[k] for k in z.files}
    T=json.loads((L/'preds/ip_theta_v1.json').read_text())
    return H.apply(F['p6'],{**H.score(M,F),'E_gr':H.lrp(gr,np.c_[F['low'],F['emb']])},T['theta'],T['theta_rc'],F['rc'])
