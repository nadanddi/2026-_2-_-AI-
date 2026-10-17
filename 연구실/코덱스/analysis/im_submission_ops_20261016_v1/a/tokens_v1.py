"""Fixed CLIP B/16 tokens; preserves the original CLS output."""
import sys,json,csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
B=Path.home()/'storage/nadan/farmai_img';PK=B/'submit_img09/팜모니_나단9';L=B/'data/local'
sys.path.insert(0,str(PK));from vit import ViTL14_336
from safetensors.torch import load_file
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def grid(w,h):
    return [(int(x),int(y)) for y in np.linspace(0,h-224,int(np.ceil((h-224)/112))+1).round().astype(int) for x in np.linspace(0,w-224,int(np.ceil((w-224)/112))+1).round().astype(int)]
def label_cells(mask,w,h):
    assert mask.shape==(h,w)
    return np.stack([mask[y:y+224,x:x+224].reshape(14,16,14,16).mean(axis=(1,3)).ravel() for x,y in grid(w,h)])
class Tokens:
    def __init__(self):
        assert torch.cuda.is_available()
        self.dev=torch.device('cuda:0');self.dt=torch.float16
        self.model=ViTL14_336(d=768,layers=12,heads=12,patch=16,image=224,out=512)
        self.model.load_state_dict({k:v.float() for k,v in load_file(str(PK/'model/b16_visual_fp16.safetensors')).items()})
        self.model=self.model.eval().to(self.dev,self.dt)
        self.mean=torch.tensor([.48145466,.4578275,.40821073],device=self.dev).view(1,3,1,1)
        self.std=torch.tensor([.26862954,.26130258,.27577711],device=self.dev).view(1,3,1,1)
        self.cached=None;self.hook=None
    def capture(self,module,inp,out):
        assert out.ndim==3 and out.shape[1:]==(197,768)
        self.cached=out[:,1:,:].detach()
    def enable(self):self.hook=self.model.ln_post.register_forward_hook(self.capture)
    def input(self,p):
        with Image.open(p) as im:a=np.asarray(im.convert('RGB'));w,h=im.size
        pcs=np.stack([a[y:y+224,x:x+224] for x,y in grid(w,h)])
        assert len(pcs)==54
        x=torch.from_numpy(pcs).to(self.dev).permute(0,3,1,2).float()/255
        return ((x-self.mean)/self.std).to(self.dt),w,h
    @torch.no_grad()
    def alignment(self,p):
        x,w,h=self.input(p);base=self.model(x).detach();self.enable();got=self.model(x).detach()
        assert torch.equal(base,got) and self.cached.shape==(54,196,768)
        synthetic=np.zeros((224,224),np.float32);synthetic[:16,:16]=1
        assert np.array_equal(label_cells(synthetic,224,224)[0],np.r_[1,np.zeros(195)])
        synthetic[:]=0;synthetic[16:32,16:32]=1
        assert label_cells(synthetic,224,224)[0,15]==1 and label_cells(synthetic,224,224).sum()==1
        return dict(cls_bitwise_equal=True,tokens=[54,196,768],mask_cell_order_row_major=True)
    @torch.no_grad()
    def sample(self,p):
        x,w,h=self.input(p);self.model(x)
        return self.cached.float().cpu().numpy(),w,h
    @torch.no_grad()
    def max_logits(self,p,W,bias):
        x,w,h=self.input(p);self.model(x)
        z=self.cached.float().reshape(-1,768)@W+bias
        assert torch.isfinite(z).all()
        return z.max(0).values.cpu().numpy()
