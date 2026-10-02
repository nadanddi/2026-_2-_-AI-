from pathlib import Path
import csv,json,math,collections,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=ROOT/'집/코덱스/local/temp_diverse_residual_20261003_v1'
def score(errors):return math.sqrt(math.fsum(x*x for x in errors)/len(errors))
def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'));groups=collections.defaultdict(list)
    with (OUT/'oof.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):groups[(row['member'],row['validator'],int(row['seed']),row['context'])].append(row)
    with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels={r['row_id']:float(r['sub_temp']) for r in csv.DictReader(f)}
    z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));zi={str(r):i for i,r in enumerate(z['row_id'])};checks=[];diagnostics=[];rng=np.random.default_rng(20261003)
    for key,rows in sorted(groups.items()):
        tag,name,seed,context=key;assert len({r['row_id'] for r in rows})==len(rows)
        assert all(labels[r['row_id']]==float(r['sub_temp']) for r in rows)
        ea=[float(r['base'])-float(r['sub_temp']) for r in rows];eb=[float(r['candidate'])-float(r['sub_temp']) for r in rows]
        a,b=score(ea),score(eb);stored=next(s for s in result['summary'] if (s['member'],s['validator'],s['seed'],s['context'])==key)
        assert abs(a-stored['baseline_rmse'])<1e-12 and abs(b-stored['candidate_rmse'])<1e-12
        item=dict(member=tag,validator=name,seed=seed,context=context,n=len(rows),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1))
        pfn=np.load(Path(env.LOCAL)/f'web_tabpfn_{"v2" if context=="1-8" else "v6"}_temp_{name}.npy')[:8].mean(0);bs={726:7,727:101}[seed];cache={};blendmax=0.
        for row in rows:
            i=zi[row['row_id']];k=int(row['fold'])
            if k not in cache:
                c=dict(np.load(OUT/f'{name}_{k}.npz',allow_pickle=True));cache[k]=(c,{str(r):j for j,r in enumerate(c['row_id'])})
            c,ci=cache[k];j=ci[row['row_id']];new=float(c[f'{tag}_{seed}'][j]);assert abs(new-float(row['new_member']))<1e-12
            t=float(row['in_temp']) if row['in_temp'] else math.nan;g=1. if math.isnan(t) else max(0.,min(1.,(t-8)/2))
            base=float(z[f'{name}__MASK__{bs}'][i]);cx=float(z[f'{name}__CODEX__{seed}'][i]);ref=math.fsum([(.4+.1*g)*base,(.6-.4*g)*cx,.3*g*float(pfn[i])]);cand=math.fsum([(.4+.0*g)*base,(.6-.4*g)*cx,.3*g*float(pfn[i]),.1*g*new])
            blendmax=max(blendmax,abs(ref-float(row['base'])),abs(cand-float(row['candidate'])))
            if g==0:assert float(row['base'])==float(row['candidate'])
        assert blendmax<1e-12;item['blend_maxdiff']=blendmax
        if name=='DIAG10':
            blocks=collections.defaultdict(list)
            for r,x,y in zip(rows,ea,eb):blocks[r['farm']+'_'+str(int(r['day'])//5)].append(y*y-x*x)
            sums=np.array([math.fsum(blocks[k]) for k in sorted(blocks)]);counts=np.array([len(blocks[k]) for k in sorted(blocks)]);ix=rng.integers(0,len(sums),size=(20000,len(sums)));means=sums[ix].sum(1)/counts[ix].sum(1);pw=float(np.mean(means>=0));ci=np.quantile(means,[.00625,.99375]).tolist()
            assert pw==stored['p_worse'] and np.allclose(ci,stored['ci'],rtol=0,atol=1e-12);item.update(p_worse=pw,ci=ci)
            days=collections.defaultdict(list)
            for r,x,y in zip(rows,ea,eb):days[(r['farm'],int(r['day']))].append((x,y,r))
            level0=math.fsum(len(v)*(math.fsum(a for a,b,r in v)/len(v))**2 for v in days.values())
            level1=math.fsum(len(v)*(math.fsum(b for a,b,r in v)/len(v))**2 for v in days.values())
            sse0=math.fsum(x*x for x in ea);sse1=math.fsum(x*x for x in eb)
            shape0=math.fsum(math.fsum((a-math.fsum(aa for aa,bb,rr in v)/len(v))**2 for a,b,r in v) for v in days.values())
            shape1=math.fsum(math.fsum((b-math.fsum(bb for aa,bb,rr in v)/len(v))**2 for a,b,r in v) for v in days.values())
            assert abs(sse0-level0-shape0)<1e-9 and abs(sse1-level1-shape1)<1e-9
            diag=dict(member=tag,seed=seed,context=context,days=len(days),baseline_level_sse_pct=100*level0/sse0,candidate_level_sse_pct=100*level1/sse1,baseline_level_rmse=math.sqrt(level0/len(rows)),candidate_level_rmse=math.sqrt(level1/len(rows)),baseline_shape_rmse=math.sqrt(shape0/len(rows)),candidate_shape_rmse=math.sqrt(shape1/len(rows)))
            em=np.array([float(r['new_member'])-float(r['sub_temp']) for r in rows]);diag['error_corr_new_baseline']=float(np.corrcoef(em,np.array(ea))[0,1]);diagnostics.append(diag)
            predicates={'late':lambda r:int(r['day'])>=179,'early':lambda r:int(r['day'])<179,'F13':lambda r:r['farm']=='F13','F47':lambda r:r['farm']=='F47','cold<=8':lambda r:bool(r['in_temp']) and float(r['in_temp'])<=8,'warm>=10':lambda r:bool(r['in_temp']) and float(r['in_temp'])>=10,'hour00_05':lambda r:int(r['hour'])<6,'hour06_17':lambda r:6<=int(r['hour'])<18,'hour18_23':lambda r:int(r['hour'])>=18}
            for segment,fn in predicates.items():
                ix=[i for i,r in enumerate(rows) if fn(r)];sa=score([ea[i] for i in ix]);sb=score([eb[i] for i in ix]);diagnostics.append(dict(member=tag,seed=seed,context=context,segment=segment,n=len(ix),baseline_rmse=sa,candidate_rmse=sb,delta_pct=100*(sb/sa-1)))
        checks.append(item)
    verdicts={tag:('PASS' if all(s['delta_pct']<0 for s in checks if s['member']==tag) and all(s['p_worse']<.00625 and s['ci'][1]<0 for s in checks if s['member']==tag and s['validator']=='DIAG10') else 'REJECT') for tag in ['CB','ET']}
    assert verdicts==result['verdicts'];assert result['cache_maxdiff']<1e-8
    verified=dict(status='PASS',verdicts=verdicts,cells=checks,diagnostics=diagnostics,cache_maxdiff=result['cache_maxdiff']);(HERE/'verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 온도 W30G 새 잔차 모델 결과 · 2026-10-03','','물리 Ridge baseline을 유지하고 새로운 잔차 모델을 따뜻한 행에 10% 섞었다. 비교 모델과 판정은 main95c0792/5d03e65에 학습 전에 고정했다.','','| 새 잔차 모델 | DIAG10 변화 | EXT10 변화 | EXT12 변화 | DIAG p_worse | 판정 |','|---|---|---|---|---|---|']
    for tag in ['CB','ET']:
        ranges=[]
        for val in ['DIAG10','EXT10','EXT12']:
            ds=[x['delta_pct'] for x in checks if x['member']==tag and x['validator']==val];ranges.append(f'{min(ds):+.3f}~{max(ds):+.3f}%')
        ps=[x['p_worse'] for x in checks if x['member']==tag and x['validator']=='DIAG10'];lines.append(f'| {tag} | '+' | '.join(ranges)+f' | {min(ps):.5f}~{max(ps):.5f} | {verdicts[tag]} |')
    lines+=['','## 검산 근거','- DIAG10 400일9600행, EXT10/EXT12 저온일 전체 홀드아웃·±1일buffer. 2시드×2문맥×3검증기=후보별12칸. 8도이하 예측변화0. 통계기준은 이번세션계절2안도포함한k4 p<.00625 및98.75% CI상한<0.','- 원시train_y/행유일성/RMSE fsum·NumPy/멤버가중합/블록bootstrap 재계산PASS. BASE/PFN 고정캐시, 원 CODEX재학습캐시비교 최대차는 verification.json. 새physics+잔차 train/validation RMSE는 result.json.','- 하루 수준/모양 오차 분해 및후반·온실·시간대·저온결과는 verification.json diagnostics. 이정보는사후진단용이며 비중/게이트재선택에쓰지않음.','','## 해석의 범위','- 기각이라면 이번 고정 모델설정·10% 온도게이트 결합을 기각한 것이다. 모든 CatBoost/ExtraTrees 설정이나 더 나은 모델 가능성을 부정하지 않는다.','- 성능이좋아도 단독수치/오차상관만으로채택하지않고 전검증기방향·통계규칙을요구한다. 두안이실패하면 이번 새멤버추가 실험을 종료한다.','- 학습과검증의격차는새physics+잔차 구성원이며 전체W30G학습오차가아니다. 공개검증반복사용·과거전체학습일기반가중치rank·PFN캐시재사용한계, 독립미사용온도홀드아웃/공식평가점수없음.','- 입력MASK, 기존인과reset특징보존, fold별Ridge/중앙값대체fit. EC계절표/잠금파일/정답읽기0, 평가예측/제출물/플랫폼업로드없음.','- 재현·산술 신뢰도 높음; 숨은평가일 일반화는확인되지않음. 구별되는새정보를찾아야하며 사후정답집단으로보정규칙을만들면안됨.','','## 파일','- run_v2.py (기존EC CatBoost 패키지경로wrapper), run.py, PROTOCOL.md, ENV_REPAIR.md. 최초실행import실패에서fit0, 모델변경없음.','- local/temp_diverse_residual_20261003_v1의fold별NPZ/oof.csv/result.json. verification.json/result.json은작은Git사본,큰결과는Drive동기화.']
    (HERE/'결과보고서_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',verdicts=verdicts,summary=checks),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

