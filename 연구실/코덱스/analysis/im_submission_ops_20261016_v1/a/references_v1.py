"""Check baseline row correspondence before candidate scores exist."""
import json
from pathlib import Path
import numpy as np
from tokens_v1 import L,read
from eval_support_v1 import metadata
HERE=Path(__file__).resolve().parent
sets=('im_set_newreal','im_set_realg','im_set_realn','im_set_r2','im_set_r3','im_set_r4','im_set_r5','sealed_real','sealed_brown','sealed_bright','jig500','pub21','im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444','im_set_gpt1','im_set_gpt3','im_set_gptr','im_set_gptr_src')
counts={}
for sn in sets:
    rr,ps=metadata(sn)
    with np.load(L/'feats_h'/f'{sn}.npz') as z:assert z['files'].tolist()==[r['file'] for r in rr]
    assert all(p.is_file() for p in ps)
    if sn in ('im_set_gptr','im_set_gptr_src'):rr=[r for r in rr if r['split']=='hold']
    y=np.array([int(r['label']) for r in rr])
    if sn=='im_set_r5':
        with np.load(HERE.parents[1]/'im_m2_20261016_v1/open/eval_im_set_r5.npz') as z:
            assert np.array_equal(y,z['y']) and np.array_equal([r['kind'] for r in rr],z['kind']) and np.array_equal([r['group'] for r in rr],z['group'])
    else:
        p=HERE.parents[1]/'im_meta_probe_20261016_v1'/f'pred_1601_{sn}.csv';assert p.is_file()
        br=read(p);assert [r['file'] for r in rr]==[r['file'] for r in br]
        assert np.array_equal(y,[int(r['label']) for r in br])
    counts[sn]=len(rr)
with (HERE/'references_v1.json').open('x') as f:json.dump(counts,f,indent=2)
print('ALL_REFERENCE_ROWS_OK',counts,flush=True)
