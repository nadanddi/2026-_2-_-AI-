from pathlib import Path
import sys,json,csv,math,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ar_float(vals):
    # Match declared ndarray(float64) SHA with little-endian doubles; no external y read.
    import struct
    return hashlib.sha256(b'float64'+str((len(vals),)).encode()+b''.join(struct.pack('<d',v) for v in vals)).hexdigest()
p=json.loads((ROOT/'집/코덱스/analysis/ec_actual_A_nested_oof_20261006_v1/preparation_v4.json').read_text(encoding='utf-8'))
src=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'
with src.open(encoding='utf-8-sig',newline='') as f:
    public={r['row_id']:float(r['y']) for r in csv.DictReader(f) if r['validator']=='DIAG10'}
assert len(public)==8640
proof=json.loads((L/'nested_snapshot/manifest_verification_v2.json').read_text(encoding='utf-8'));expected={Path(z['path']).name:z for z in proof['files']};checked=[]
for k in range(4):
    entries=[r for r in p['records'] if r['v']=='DIAG10' and r['k']==k];assert {r['j'] for r in entries}==set(range(4))
    outer=entries[0]['outer_train_ids'];hold=set(entries[0]['outer_query_ids'])
    assert all(r['outer_train_ids']==outer and set(r['outer_query_ids'])==hold for r in entries)
    for e in entries:
        assert not (set(e['train_ids'])|set(e['query_ids']))&hold
        assert ar_float([public[i] for i in e['train_ids']])==e['train_target_sha']
        assert ar_float([public[i] for i in e['query_ids']])==e['query_target_sha']
        bans={(i[:3],int(i[4:7])+j) for i in e['query_ids'] for j in [-1,0,1]}
        assert not {(i[:3],int(i[4:7])) for i in e['train_ids']}&bans
    for s in [7,101,2024]:
        name=f'OOF_DIAG10_{k}_{s}.csv';path=L/'nested_snapshot'/name;assert sha(path)==expected[name]['sha256']
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        assert [r['row_id'] for r in rows]==outer and len(set(outer))==len(outer)
        for r in rows:
            assert int(r['k'])==k and int(r['s'])==s and r['v']=='DIAG10'
            assert abs(float(r['y'])-public[r['row_id']])<1e-12
            e=next(e for e in entries if e['j']==int(r['j']));assert r['row_id'] in set(e['query_ids'])
            raw=.8*float(r['raw_r3'])+.2*float(r['raw_pfn']);assert abs(raw-float(r['raw_A']))<1e-12
            assert [float(r['clip_lo']),float(r['clip_hi'])]==e['bounds']
        days={}
        for r in rows:days.setdefault((r['farm'],int(r['day'])),[]).append(r)
        for key,g in days.items():
            g=sorted(g,key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24));total=0;ptotal=0;ym=math.fsum(float(r['y']) for r in g)/24
            for h,r in enumerate(g):
                total+=float(r['raw_A']);a=max(float(r['clip_lo']),min(float(r['clip_hi']),.5*float(r['raw_A'])+.5*total/(h+1)))
                assert abs(a-float(r['A']))<1e-12
                ptotal+=a;assert abs(ptotal/(h+1)-float(r['prefix_A']))<1e-12
                assert abs(ym-float(r['y_day']))<1e-12
        checked.append(dict(k=k,s=s,sha=sha(path),rows=len(rows),train_days=len(days),outer_query_rows=len(hold)))
out=dict(status='PASS_PARTIAL_NESTED_INPUTS',files=checked,outer_folds=4,inner_contexts=16,full80=False,fit=0,raw_y_reads=0,test_reads=0)
(H/'nested_input_audit_v1.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
