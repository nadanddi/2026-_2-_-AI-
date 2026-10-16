#!/bin/bash
set -eu
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
from pathlib import Path
import csv, numpy as np
B=Path.home()/'storage/nadan/farmai_img';L=B/'data/local'
for sn in ['im_set_gpt1','im_set_gpt2','im_set_gpt3','im_set_gptr','im_set_gptr_src','im_set_h3','im_set_h3j']:
 p=L/sn/'manifest.csv';r=list(csv.DictReader(open(p,encoding='utf-8')))
 print(sn,len(r),'fields',list(r[0]),'examples',r[:2],flush=True)
print('preds', [p.name for p in (L/'preds').glob('*') if any(k in p.name for k in ['oof','m_v2','ip_'])])
print('ownM2',list((Path.home()/'storage/nadan/codex/im_m2_20261016_v1').glob('**/*oof*')))
PY
if [ ! -f ~/storage/nadan/farmai_img/data/local/feats_h/im_set_gpt2.npz ]; then
 nvidia-smi
 python -u ~/storage/nadan/farmai_img/data/image/im43_h_feats.py im_set_gpt2
fi
