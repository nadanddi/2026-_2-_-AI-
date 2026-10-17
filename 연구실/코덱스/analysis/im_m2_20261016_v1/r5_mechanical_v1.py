"""R5 production integrity only: never reads cached score arrays or reports accuracy."""
import sys,csv,json,hashlib,tarfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def main(mode):
    if mode=='extract':
        assert not (L/'im_r5_raw').exists()
        with tarfile.open(HERE/'im_r5_raw_v1.tar') as tf:
            for m in tf.getmembers():
                p=(L/m.name).resolve()
                assert p.is_relative_to((L/'im_r5_raw').resolve())
                assert m.isfile() or m.isdir()
            tf.extractall(L)
        reg=json.loads((HERE/'registration_v1.json').read_text(encoding='utf-8-sig'))
        assert sha(L/'im_r5_raw/manifest.csv')==reg['manifest_sha256']
        inv=json.loads((HERE/'raw_inventory_v1.json').read_text(encoding='utf-8'))
        for r in inv:
            p=L/'im_r5_raw'/r['path']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        with (HERE/'raw_transfer_v1.json').open('x') as f: json.dump(dict(images=len(inv),all_sha_equal=True),f)
        print('RAW_TRANSFER_PASS',len(inv),flush=True); return
    assert mode in ('images','features')
    sys.path.insert(0,str(B/'data/image'))
    import im64_make_r5 as M
    jobs,_=M.plan(); rr=rows(L/'im_set_r5/manifest.csv')
    assert len(jobs)==len(rr)==700
    expected={sid+('_real.jpg' if real else '_fake.jpg'):(r,real,kind) for _,sid,r,real,kind in jobs}
    assert len(expected)==700 and len({r['file'] for r in rr})==700
    assert {r['file'] for r in rr}==set(expected)
    assert [r['file'] for r in rr]==sorted(expected)
    for r in rr:
        raw,real,kind=expected[r['file']]
        assert r['source']==raw['file'] and r['group']==raw['group']
        assert int(r['label'])==(0 if real else 1)
        assert r['kind']==('real' if real else kind)
        assert (L/'im_set_r5/img'/r['file']).is_file()
    assert len(list((L/'im_set_r5/img').glob('*.jpg')))==700
    # Provenance only, not scoring. External check runs before R5 evaluation.
    tr=[]
    for sn in ('im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src'):
        tr.extend(r for r in rows(L/sn/'manifest.csv') if r.get('split','train')=='train')
    sd=not ({r['source'] for r in rr}&{r['source'] for r in tr})
    gd=not ({r['group'] for r in rr}&{r['group'] for r in tr})
    assert sd and gd
    record=dict(complete=True,images=700,source_disjoint=sd,group_disjoint=gd,features_ok=False)
    if mode=='features':
        p=L/'feats_h/im_set_r5.npz'
        with np.load(p) as z:
            # Deliberately load only these two metadata arrays. No p6/ppre/rc/ff/embedding access.
            assert z['files'].tolist()==[r['file'] for r in rr]
            assert len(z['ok'])==700 and z['ok'].all()
        record.update(features_ok=True,feature_sha256=sha(p),manifest_sha256=sha(L/'im_set_r5/manifest.csv'))
        with (HERE/'r5_integrity_v1.json').open('x') as f: json.dump(record,f,indent=2)
    print('R5_MECHANICAL_PASS',mode,700,'source/group disjoint; no scores read',flush=True)
if __name__=='__main__': main(sys.argv[1])
