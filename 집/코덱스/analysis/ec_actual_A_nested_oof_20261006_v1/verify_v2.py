from pathlib import Path
import sys,json,csv,math,hashlib,collections,argparse,zipfile
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env;import env_extra
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    prep=load(H/'preparation_v2.json');reg=load(H/'registration_v2.json')
    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name
    for path,digest in prep['dependencies'].items():assert sha(ROOT/path)==digest,path
    assert sha(Path(env.DATA)/'train_X.csv')==prep['inputs']['train_X']
    public_path=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
    assert sha(public_path)==prep['inputs']['public_oof']
    public={r['row_id']:float(r['sub_ec']) for r in rows(public_path) if r['validator']=='DIAG10' and int(r['seed'])==7};assert len(public)==8640
    expected_contexts=prep['records'];completed=0;component_count=0;rawgap=0.;repeat=[];contexts={}
    for e in expected_contexts:
        v,k,j=e['v'],e['k'],e['j'];prefix=f'{v}_{k}_{j}';contexts[v,k,j]=e
        tri=e['train_ids'];qi=e['query_ids'];outer=e['outer_query_ids'];qset=set(qi);tset=set(tri)
        assert len(tri)==len(tset) and len(qi)==len(qset) and not tset&qset
        assert not (tset|qset)&set(outer);assert tset|qset<=set(e['outer_train_ids'])
        qdays={(rid[:3],int(rid[4:7])) for rid in qi};banned={(f,d+o) for f,d in qdays for o in [-1,0,1]}
        assert not {(rid[:3],int(rid[4:7])) for rid in tri}&banned
        names=[(f'{prefix}_r3_{s}.npz','r3',s) for s in [7,101,2024]]+[(f'{prefix}_pfn_{s}.npz','pfn',s) for s in [1,2,3,4]];done=0
        for name,kind,s in names:
            path=OUT/name
            if not path.exists() or not path.with_suffix('.json').exists():assert a.partial;continue
            d=arrays(path);m=load(path.with_suffix('.json'));assert m['sha']==sha(path)
            assert d['row_id'].tolist()==qi and d['train_row_id'].tolist()==tri
            assert np.isfinite(d['raw']).all() and len(d['raw'])==len(qi)
            assert m['signature']['runtime']==prep['runtime'] and m['signature']['dependencies']==prep['dependencies'];assert m['signature']['kind']==('R3 .6ET+.3LGB+.1MLP' if kind=='r3' else 'TabPFN V2 CPU float32 context2000 estimator4')
            assert m['signature']['prepared_sha']==sha(H/'preparation_v2.json') and m['signature']['seed']==s
            record_sha=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest();assert m['signature']['record_sha']==record_sha
            if kind=='pfn':
                ix=np.random.default_rng(s).choice(len(tri),min(2000,len(tri)),replace=False);assert d['context_row_id'].tolist()==[tri[int(i)] for i in ix]
                assert not set(d['context_row_id'])&(qset|set(outer));assert m['details']['first8_batch_invariance']
            else:
                independent=np.asarray([math.fsum([.6*float(x),.3*float(y),.1*float(z)]) for x,y,z in zip(d['et'],d['lgb'],d['mlp'])]);rawgap=max(rawgap,float(np.max(abs(independent-d['raw']))));assert rawgap<1e-12
            if m['details']['repeat_maxdiff'] is not None:repeat.append(dict(context=[v,k,j],kind=kind,seed=s,maxdiff=m['details']['repeat_maxdiff']))
            component_count+=1;done+=1
        completed+=int(done==7)
    if a.partial:
        print(json.dumps(dict(status='PASS_PARTIAL_AUDIT_ONLY',complete_contexts=completed,total_contexts=80,checked_components=component_count,raw_mix_maxdiff=rawgap,repeats=repeat),ensure_ascii=False));return
    assert completed==80 and component_count==560
    assert {(z['kind'],z['seed']) for z in repeat}=={('r3',7),('pfn',1)}
    assert all(z['maxdiff']<(1e-9 if z['kind']=='r3' else 1e-5) for z in repeat)
    receipt=load(H/'receipt_v2.json');assert receipt['status']=='COMPLETE_80_CONTEXTS_ACTUAL_A_OOF'
    for f in receipt['files']:assert sha(OUT/f['path'])==f['sha']
    groups=collections.defaultdict(list);coverage=[];gaps=dict(A=0.,prefix_A=0.,raw_A=0.,PFN=0.)
    case_expected={};summary=collections.defaultdict(collections.Counter)
    for v,k in dict.fromkeys((e['v'],e['k']) for e in expected_contexts):
        records=[e for e in expected_contexts if e['v']==v and e['k']==k];assert len(records)==4
        outertr=records[0]['outer_train_ids'];assert all(e['outer_train_ids']==outertr for e in records)
        assert sorted(rid for e in records for rid in e['query_ids'])==sorted(outertr)
        for s in [7,101,2024]:
            data=rows(OUT/f'OOF_{v}_{k}_{s}.csv');assert [r['row_id'] for r in data]==outertr
            indexed={r['row_id']:r for r in data};assert len(indexed)==len(data)
            for e in records:
                j=e['j']
                yy=np.asarray([float(indexed[rid]['y']) for rid in e['query_ids']],dtype=float)
                target_hash=hashlib.sha256(str(yy.dtype).encode()+str(yy.shape).encode()+yy.tobytes()).hexdigest();assert target_hash==e['query_target_sha']
                assert max(abs(float(indexed[rid]['y'])-public[rid]) for rid in e['query_ids'])<1e-12
                z=arrays(OUT/f'{v}_{k}_{j}_r3_{s}.npz');bags=[arrays(OUT/f'{v}_{k}_{j}_pfn_{c}.npz')['raw'] for c in [1,2,3,4]]
                for i,rid in enumerate(e['query_ids']):
                    r=indexed[rid];pfn=math.fsum(float(b[i]) for b in bags)/4;raw=.8*float(z['raw'][i])+.2*pfn
                    assert int(r['j'])==j and [float(r['clip_lo']),float(r['clip_hi'])]==e['bounds']
                    gaps['PFN']=max(gaps['PFN'],abs(pfn-float(r['raw_pfn'])));gaps['raw_A']=max(gaps['raw_A'],abs(raw-float(r['raw_A'])))
                    groups[(v,k,s,r['farm'],int(r['day']))].append(r)
            coverage.append(dict(v=v,k=k,s=s,rows=len(data),days=len(data)//24))
    for key,g in groups.items():
        g.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24));assert len({r['j'] for r in g})==1
        yy=math.fsum(float(r['y']) for r in g)/24;history=[];ah=[]
        for r in g:
            raw=float(r['raw_A']);history.append(raw);pred=min(float(r['clip_hi']),max(float(r['clip_lo']),.5*raw+.5*math.fsum(history)/len(history)));ah.append(float(r['A']));prefix=math.fsum(ah)/len(ah)
            gaps['A']=max(gaps['A'],abs(pred-float(r['A'])));gaps['prefix_A']=max(gaps['prefix_A'],abs(prefix-float(r['prefix_A'])))
            flags=dict(high=int(yy>=1),hard_high=int(yy>=1 and prefix<1.2),missed_high=int(yy>=1 and prefix<.9),hard_low=int(yy<1 and prefix>=.9))
            for label,value in flags.items():assert int(r[label])==value
            assert abs(float(r['y_day'])-yy)<1e-12
            if int(r['hour']) in [0,6,12,23]:
                case_expected[(r['v'],int(r['k']),int(r['s']),r['row_id'])]=r
                z=summary[(r['v'],int(r['k']),int(r['s']),int(r['hour']))];z.update(n=1,high=flags['high'],hard_high=flags['hard_high'],missed_high=flags['missed_high'],hard_low=flags['hard_low'])
    assert max(gaps.values())<1e-12 and len(coverage)==60,gaps
    c=rows(OUT/'nested_cases_v1.csv');assert len(c)==len(case_expected)
    for r in c:
        key=(r['v'],int(r['k']),int(r['s']),r['row_id']);assert r==case_expected.pop(key)
    assert not case_expected
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
    proof=dict(status='PASS_COMPLETE_ACTUAL_A_NESTED_OOF',complete_contexts=completed,components=component_count,outer_oof_files=len(coverage),total_hourly_occurrences=sum(z['rows'] for z in coverage),total_case_occurrences=len(c),raw_mix_maxdiff=rawgap,maxdiff=gaps,repeats=repeat,coverage=coverage,case_counts=[dict(v=v,k=k,s=s,hour=h,**z) for (v,k,s,h),z in summary.items()],limitations=['Repeated public validation, not a new independent holdout','Each outer fold is a separate classifier training context; do not pool/re-split globally','No new classifier or expert mixture performance measured'])
    if a.check_only:
        assert proof==load(H/'verification_v2.json')
        print(json.dumps(dict(status='PASS_FRESH_RECHECK_NO_OVERWRITE',complete_contexts=completed,total_hourly_occurrences=proof['total_hourly_occurrences'],maxdiff=gaps),ensure_ascii=False));return
    save(H/'verification_v2.json',proof)
    description=['# 실제 A의 외부 학습 집합 내부 교차 예측 — 자동 검산 기록','','80개 내부 분할에서 R3 240개와 TabPFN 320개를 적합하고 각 외부 학습 날짜를 정확히 한 번 예측했다. 외부 DIAG10·A·B 20개 × 세 시드의 학습용 OOF CSV 60개를 저장했다. 내부 검증 날짜와 같은 농장의 ±1일을 제외했다.','',f"독립 검산 PASS: {proof['total_hourly_occurrences']:,} 시간행 출현, {len(c):,} 사례행 출현. 최대 후처리 오차 {gaps['A']:.3g}, prefix 오차 {gaps['prefix_A']:.3g}. 첫 R3/PFN 독립 재적합 및 PFN 첫8행 배치 일치도 통과.",'','모델·특징·season·TabPFN 설정·혼합·후처리는 기존 seasonv2 A와 같고, 전처리와 범위는 각 내부 학습 집합에서 정했다. 새 분류기를 학습한 결과가 아니며 RMSE 개선을 주장하지 않는다. 이 문서는 자동 산술 검산 기록으로, 사용자에게 결과를 확정하여 전달하기 전 혹독한 비평가의 평가와 문제별 개선책 피드백을 추가해야 한다.','', '학습용 OOF는 각 외부 fold의 기존 외부 검증 A 예측과 짝지어 사용해야 한다. 여러 외부 fold의 OOF를 합쳐 재분할하면 누수가 생길 수 있다.','', '재현: 등록된 run_v2.py 실행. preparation_v2.json과 구성원 NPZ/JSON을 검증한 뒤 기존 완료 캐시를 재사용한다. 정상 완료 뒤 다시 전체 실행하지 말고 verify_v2.py --check-only로 검산한다. 비정상 종료 lock을 자동 삭제하지 않는다.','', '재현 ZIP에는 코드·설정·검산·구성원 예측 캐시·OOF CSV를 포함했다. TabPFN checkpoint는 기존 로컬 파일을 SHA256으로 검증하며 ZIP에 넣지 않았다. 실제 학습 모델 객체는 저장하지 않으며 동일 코드·시드·환경으로 다시 적합한다.','', '운영위 원 PDF 원문 미확보; 저장소 안내 요약에 기반. 잠금 EL1/평가 정답/평가 전체 입력 통계 및 리더보드로 계수를 정하지 않았다.']
    with (H/'학습산출물_검산_v2.md').open('x',encoding='utf-8') as f:f.write('\n'.join(description)+'\n')
    zp=OUT/'reproduction_v2.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['run_v2.py','verify_v2.py','plan_v1.md','registration_v2.json','preparation_v2.json','receipt_v2.json','verification_v2.json','학습산출물_검산_v2.md']:z.write(H/name,'analysis/'+name)
        for f in receipt['files']:
            p=OUT/f['path'];z.write(p,'local/'+p.name)
            if p.suffix=='.npz':z.write(p.with_suffix('.json'),'local/'+p.with_suffix('.json').name)
    save(H/'bundle_v2.json',dict(path=zp.name,sha=sha(zp)))
    print(json.dumps({k:proof[k] for k in ['status','complete_contexts','components','outer_oof_files','total_hourly_occurrences','total_case_occurrences','maxdiff']},ensure_ascii=False))
if __name__=='__main__':main()
