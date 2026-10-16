"""Verify saved actual arrays, build identities and complete candidate artifacts."""
import tarfile,io,json,hashlib,ast
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3];L=ROOT/'연구실/코덱스/local/im_n10_n11_prepare_20261016_v1'
builds={10:json.loads((P/'build_result_upper_v1.json').read_text(encoding='utf-8'))['10'],11:json.loads((P/'build_result_v2.json').read_text(encoding='utf-8'))['11']}
checks=0;actual_rows=0
source=(P/'inference_n9_source.py').read_text(encoding='utf-8')
tree=ast.parse((P/'build_upper_v1.py').read_text(encoding='utf-8'))
alarm=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='alarm' for t in node.targets))
assert (P/'inference_n10_upper_v1.py').read_text(encoding='utf-8')==source.replace('    return min(max(p, 0.0), 1.0)',alarm+'    return min(max(p, 0.0), 1.0)')
assert (P/'inference_n11_v1.py').read_text(encoding='utf-8')==source.replace(' and p < 0.8: p = min(p, 0.3); rc = True',': p = min(p, 0.3); rc = True')
checks+=2
with tarfile.open(L/'actual_checks_v1.tar') as tf:
    for member in tf.getmembers():
        if not member.name.endswith('.npz'):continue
        # Retain the approved upper N10 and the unchanged N11 arrays only.
        take=('build_upper_v1/n10/actual_upper_' in member.name) or ('build_v2/n11/actual_' in member.name and 'actual_upper_' not in member.name)
        if not take:continue
        with np.load(io.BytesIO(tf.extractfile(member).read())) as z:
            actual=z['actual'];exp=z['expected'];yy=z['y'];assert np.isfinite(actual).all() and ((0<=actual)&(actual<=1)).all()
            assert np.array_equal(actual>=.5,exp>=.5);checks+=2;actual_rows+=len(yy)
            if 'ff' in z:
                ff=z['ff'];alarm=np.isfinite(ff).all(1)&(ff[:,1]>=3.4739584922790527)&(ff[:,0]>1.0911458730697632)&(ff[:,0]<2.3890950679779053)
                assert np.array_equal(actual,np.maximum(z['baseline_actual'],np.where(alarm,.6,0)));checks+=1
for n,b in builds.items():
    arc=L/f'팜모니_나단{n}.zip';assert hashlib.sha256(arc.read_bytes()).hexdigest()==b['zip_sha256'] and arc.stat().st_size<1e9 and b['size']<1e9;checks+=3
for name,n in (('self_checks_upper_v1.json','10'),('self_checks_v1.json','11')):
    r=json.loads((P/name).read_text(encoding='utf-8'))[n];assert not r['errors'] and r['info']['sample_metrics']['accuracy']==1 and r['info']['run']['order_check']['consistent'];checks+=3
out=dict(pass_all=True,checks=checks,actual_rows=actual_rows,N10_upper=True,N11_unchanged=True,E4_access=False,T4_pending=True,upload=False)
with (P/'verification_final_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('FINAL_VERIFIED',out)
