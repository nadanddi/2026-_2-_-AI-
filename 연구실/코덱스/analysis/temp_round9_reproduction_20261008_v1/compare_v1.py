"""Compare temperature only to round9, without any evaluation labels or scoring."""
from pathlib import Path
import sys,csv,json,math,hashlib
from decimal import Decimal
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'))
import numpy as np
P=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
E=json.loads((H/'execution_v1.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
out=Path(P['output']);p=out/'temp_candidate_v13.csv';npz=out/'temp_candidate_v13_members.npz'
ref=Path(P['package'])/'온도/temp_candidate_v13.csv';submitted=ROOT/'제출/09회차_2026-10-06(팀)/submission_14.csv'
a,b,c=map(read,[p,ref,submitted]);assert len(a)==len(b)==len(c)==1440
ids=[r['row_id'] for r in a];assert len(set(ids))==1440 and ids==[r['row_id'] for r in b]==[r['row_id'] for r in c]
assert all(math.isfinite(float(r['sub_temp'])) for r in a+b+c)
diff=[float(x['sub_temp'])-float(y['sub_temp']) for x,y in zip(a,c)]
changed=[i for i,(x,y) in enumerate(zip(a,c)) if Decimal(x['sub_temp'])!=Decimal(y['sub_temp'])]
refchanged=[i for i,(x,y) in enumerate(zip(a,b)) if Decimal(x['sub_temp'])!=Decimal(y['sub_temp'])]
assert [r['sub_temp'] for r in b]==[r['sub_temp'] for r in c]
with np.load(npz,allow_pickle=False) as z:
    base,codex,pf,g,pred=[z[n].copy() for n in ['base','codex','pfn_samples','gate','pred']]
assert base.shape==codex.shape==g.shape==pred.shape==(1440,) and pf.shape==(8,1440)
assert all(np.isfinite(x).all() for x in [base,codex,pf,g,pred])
formula=.4*base+(.2+.4*(1-g))*codex+.4*g*pf.mean(axis=0)
assert np.array_equal(formula,pred)
assert all(Decimal(f'{x:.6f}')==Decimal(r['sub_temp']) for x,r in zip(pred,a))
tx={r['row_id']:float(r['in_temp']) if r['in_temp'] else math.nan for r in read(Path(P['package'])/'온도/data/test_X.csv')}
gate=np.array([1 if math.isnan(tx[i]) else min(1,max(0,(tx[i]-8)/2)) for i in ids])
assert np.array_equal(gate,g)
for n,v in P['temp_package_files'].items():assert sha(Path(P['package'])/'온도'/n)==v
assert sha(P['zip'])==P['zip_sha'] and sha(submitted)==P['submission_sha']
for n,v in P['data_sha'].items():assert sha(Path(P['package'])/'온도/data'/n)==v
assert sha(H/'run_v1.py')==E['wrapper_sha']
assert sha(Path(P['temp_code'])/'make_submission_v13_temp.py')==E['source_sha']
cache={p.name:sha(p) for p in Path(P['cache']).iterdir() if p.is_file()}
assert cache.get('tabpfn-v2-regressor.ckpt')==P['checkpoint_sha']
rawdiff=pred-np.array([float(r['sub_temp']) for r in c])
R={'status':'PASS_EXACT_TEMPERATURE_6_DECIMALS' if not changed and not refchanged else 'EXECUTED_BUT_NOT_EXACT_TEMPERATURE_REPRODUCTION',
   'rows':1440,'strict_six_decimal_match':not changed and not refchanged,'changed_temperature_rows_vs_round9':len(changed),'changed_temperature_rows_vs_zip_reference':len(refchanged),
   'max_abs_csv_diff':max(map(abs,diff)),'rms_csv_diff':math.sqrt(math.fsum(d*d for d in diff)/len(diff)),
   'raw_prediction_vs_rounded_reference_max_abs':float(abs(rawdiff).max()),'mixture_formula_exact':True,'gate_matches_official_test_input':True,
   'row_order_exact':True,'new_temperature_models_run':True,'pfn_context_predictions':8,'source_preserved':True,'input_source_sha_preserved':True,
   'file_bytes_equal_zip_reference':sha(p)==sha(ref),'file_bytes_equal_round9_whole_csv':sha(p)==sha(submitted),
   'scope':'temperature only; generated auxiliary EC is original script copy of submission04, not round9 EC reproduction',
   'extra_temp_v13_checks_executed':False,'evaluation_label_access':False,'platform_score_reproduced':False,
   'python':E['python'],'versions':E['versions'],'seconds':E['seconds'],'cache_files_sha':cache,
   'files_sha':{'generated_csv':sha(p),'generated_members':sha(npz),'zip_reference_csv':sha(ref),'round9_csv':sha(submitted),'comparison_source':sha(__file__)},
   'largest_differences':[{'row_id':ids[i],'generated':a[i]['sub_temp'],'submitted':c[i]['sub_temp'],'difference':diff[i]} for i in sorted(changed,key=lambda i:abs(diff[i]),reverse=True)[:10]]}
with (H/'comparison_v1.json').open('x',encoding='utf-8') as f:json.dump(R,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(R,ensure_ascii=False,indent=2))
