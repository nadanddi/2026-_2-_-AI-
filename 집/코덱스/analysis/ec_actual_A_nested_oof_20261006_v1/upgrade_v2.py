from pathlib import Path
H=Path(__file__).resolve().parent
def write(name,text):
    with (H/name).open('x',encoding='utf-8') as f:f.write(text)
r=(H/'run_v1.py').read_text(encoding='utf-8')
r=r.replace("H/'verify_v1.py'","H/'verify_v2.py'").replace("H/'preparation_v1.json'","H/'preparation_v2.json'").replace("H/'registration_v1.json'","H/'registration_v2.json'").replace("H/'receipt_v1.json'","H/'receipt_v2.json'")
r=r.replace("df['high']=(yd>=1).astype(int)","df['y_day']=yd;df['high']=(yd>=1).astype(int)")
needle="    save(H/'receipt_v2.json',dict(status="
new="""    consensus=[]
    for (v,k,farm,day,h),g in case.groupby(['v','k','farm','day','hour']):
        assert set(g.s)==set(SEEDS) and g.high.nunique()==1
        consensus.append(dict(v=v,k=int(k),farm=farm,day=int(day),hour=int(h),high=int(g.high.iloc[0]),y_day=float(g.y_day.iloc[0]),A_prefix_min=float(g.prefix_A.min()),A_prefix_max=float(g.prefix_A.max()),hard_high_votes=int(g.hard_high.sum()),missed_high_votes=int(g.missed_high.sum()),hard_low_votes=int(g.hard_low.sum())))
    dest=OUT/'nested_seed_consensus_v1.csv';assert not dest.exists();pd.DataFrame(consensus).to_csv(dest,index=False);files.append(dict(path=dest.name,sha=sha(dest)))
"""
assert needle in r;r=r.replace(needle,new+needle);write('run_v2.py',r)
v=(H/'verify_v1.py').read_text(encoding='utf-8')
v=v.replace("ap.add_argument('--partial',action='store_true');a=ap.parse_args()","ap.add_argument('--partial',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()")
for old,newname in [('preparation_v1.json','preparation_v2.json'),('registration_v1.json','registration_v2.json'),('receipt_v1.json','receipt_v2.json'),('verification_v1.json','verification_v2.json'),('verify_v1.py','verify_v2.py'),('run_v1.py','run_v2.py'),('완료결과_v1.md','학습산출물_검산_v2.md'),('reproduction_v1.zip','reproduction_v2.zip'),('bundle_v1.json','bundle_v2.json')]:v=v.replace(old,newname)
v=v.replace("expected_contexts=prep['records'];completed", """for path,digest in prep['dependencies'].items():assert sha(ROOT/path)==digest,path
    assert sha(Path(env.DATA)/'train_X.csv')==prep['inputs']['train_X']
    public_path=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
    assert sha(public_path)==prep['inputs']['public_oof']
    public={r['row_id']:float(r['sub_ec']) for r in rows(public_path) if r['validator']=='DIAG10' and int(r['seed'])==7};assert len(public)==8640
    expected_contexts=prep['records'];completed""")
v=v.replace("assert m['signature']['prepared_sha']==sha", "assert m['signature']['runtime']==prep['runtime'] and m['signature']['dependencies']==prep['dependencies'];assert m['signature']['kind']==('R3 .6ET+.3LGB+.1MLP' if kind=='r3' else 'TabPFN V2 CPU float32 context2000 estimator4')\n            assert m['signature']['prepared_sha']==sha")
v=v.replace("j=e['j'];z=arrays", """j=e['j']
                yy=np.asarray([float(indexed[rid]['y']) for rid in e['query_ids']],dtype=float)
                target_hash=hashlib.sha256(str(yy.dtype).encode()+str(yy.shape).encode()+yy.tobytes()).hexdigest();assert target_hash==e['query_target_sha']
                assert max(abs(float(indexed[rid]['y'])-public[rid]) for rid in e['query_ids'])<1e-12
                z=arrays""")
v=v.replace("for label,value in flags.items():assert int(r[label])==value", "for label,value in flags.items():assert int(r[label])==value\n            assert abs(float(r['y_day'])-yy)<1e-12")
v=v.replace("assert not case_expected\n    proof=", """assert not case_expected
    cg=collections.defaultdict(list)
    for r in c:cg[(r['v'],int(r['k']),r['farm'],int(r['day']),int(r['hour']))].append(r)
    consensus=rows(OUT/'nested_seed_consensus_v1.csv');assert len(consensus)==len(cg)
    for r in consensus:
        g=cg.pop((r['v'],int(r['k']),r['farm'],int(r['day']),int(r['hour'])));assert sorted(int(z['s']) for z in g)==[7,101,2024]
        assert all(int(z['high'])==int(r['high']) for z in g)
        assert all(abs(float(z['y_day'])-float(r['y_day']))<1e-12 for z in g)
        for flag in ['hard_high','missed_high','hard_low']:assert int(r[flag+'_votes'])==sum(int(z[flag]) for z in g)
        for tag,fun in [('min',min),('max',max)]:assert abs(float(r['A_prefix_'+tag])-fun(float(z['prefix_A']) for z in g))<1e-12
    assert not cg
    proof=""")
v=v.replace("save(H/'verification_v2.json',proof)","""if a.check_only:
        assert proof==load(H/'verification_v2.json')
        print(json.dumps(dict(status='PASS_FRESH_RECHECK_NO_OVERWRITE',complete_contexts=completed,total_hourly_occurrences=proof['total_hourly_occurrences'],maxdiff=gaps),ensure_ascii=False));return
    save(H/'verification_v2.json',proof)""")
v=v.replace('# 실제 A의 외부 학습 집합 내부 교차 예측 완료','# 실제 A의 외부 학습 집합 내부 교차 예측 — 자동 검산 기록')
v=v.replace('정상 완료 뒤 다시 전체 실행하지 말고 verify_v2.py로 검산한다.','정상 완료 뒤 다시 전체 실행하지 말고 verify_v2.py --check-only로 검산한다.')
v=v.replace('새 분류기를 학습한 결과가 아니며 RMSE 개선을 주장하지 않는다.','새 분류기를 학습한 결과가 아니며 RMSE 개선을 주장하지 않는다. 이 문서는 자동 산술 검산 기록으로, 사용자에게 결과를 확정하여 전달하기 전 혹독한 비평가의 평가와 문제별 개선책 피드백을 추가해야 한다.')
write('verify_v2.py',v)
pre=(H/'prelaunch_v2.py').read_text(encoding='utf-8').replace('preparation_v1.json','preparation_v2.json').replace('prelaunch_v2.json','prelaunch_v3.json');write('prelaunch_v3.py',pre)
print('UPGRADE_V2_CREATED')
