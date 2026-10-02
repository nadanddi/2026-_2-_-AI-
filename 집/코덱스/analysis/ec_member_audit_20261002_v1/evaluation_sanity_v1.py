"""기존 3시드 OOF의 동일값 후보로 공통 판정/누락 게이트 검사. 모델 학습 없음."""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,math,hashlib,importlib.util,argparse
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
COMMON=ROOT/'집/코덱스/analysis/ec_model_common_20261002_v1/common.py'
CAT=ROOT/'집/코덱스/analysis/ec_catboost_20261002_v1/run_catboost.py'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,rec):
    p=HERE/name
    assert not p.exists(),f'기존 검산 결과 보호: {p}'
    p.write_text(json.dumps(rec,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rmse(y,p):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actual-catboost',type=Path)
    parser.add_argument('--output-prefix',default='evaluation_sanity_v1');args=parser.parse_args()
    common=module('readonly_common_for_sanity',COMMON);cat=module('readonly_cat_for_sanity',CAT)
    common.prepare()
    if args.actual_catboost:
        frame=pd.read_csv(args.actual_catboost);arm='catboost';cat_frame=frame
    else:
        source=pd.read_csv(common.CACHE/'oof_predictions.csv')
        fixed=['row_id','farm','day','hour','block','sub_ec','validator','validation_fold']
        parts=[]
        for seed in common.SEEDS:
            d=source[fixed].copy();d['seed']=seed;d['v2']=source['v2_'+str(seed)];d['fake_same_as_v2']=d.v2
            parts.append(d)
        frame=pd.concat(parts,ignore_index=True);arm='fake_same_as_v2'
        cat_frame=frame.assign(baseline_v2=frame.v2,catboost=frame[arm])
    result=common.evaluate(frame,[arm]);independent_scores=[]
    for s in result['scores']:
        gg=frame[frame.validator.eq(s['validator'])&frame.seed.eq(s['seed'])]
        value=rmse(gg.sub_ec,gg[s['arm']]);assert abs(value-s['rmse'])<1e-12
        independent_scores.append({'validator':s['validator'],'seed':s['seed'],'arm':s['arm'],'math_fsum_rmse':value})
    compared=[]
    for seed in common.SEEDS:
        dg=cat_frame[cat_frame.validator.eq('DIAG10')&cat_frame.seed.eq(seed)]
        cb=cat.paired_bootstrap(dg)
        cm=next(r for r in result['bootstrap'] if r['seed']==seed)
        cierr=max(abs(a-b) for a,b in zip(cb['ci_bonferroni5'],[cm['ci_rmse_low'],cm['ci_rmse_high']]))
        perror=abs(cb['p_worse']-cm['p_worse'])
        assert cierr<=1e-12 and perror<=1e-12
        compared.append({'seed':seed,'catboost_paired_bootstrap':cb,'common_bootstrap':cm,
                         'maximum_rmse_ci_difference':cierr,'p_worse_difference':perror})
    rejected=[]
    if not args.actual_catboost:
        assert len(result['scores'])==30 and len(result['bootstrap'])==3
        assert all(s['delta_rmse']==0 and s['delta_pct']==0 for s in result['scores'] if s['arm']==arm)
        assert all(b['delta_mse']==0 and b['ci_mse_low']==b['ci_mse_high']==b['ci_rmse_low']==b['ci_rmse_high']==0 and b['p_worse']==1 for b in result['bootstrap'])
        assert result['decisions'][0]['adopted'] is False and result['decisions'][0]['improving_seed_validator_cells']==0
        bad_cases={
          'missing_seed_validator_cell':frame[~(frame.validator.eq('B')&frame.seed.eq(101))],
          'missing_fold':frame[~(frame.validator.eq('DIAG10')&frame.seed.eq(7)&frame.validation_fold.eq(0))],
          'missing_single_occurrence':frame.iloc[1:].copy(),
          'duplicated_single_occurrence':pd.concat([frame,frame.iloc[[0]]],ignore_index=True)}
        for name,bad in bad_cases.items():
            try:common.evaluate(bad,[arm])
            except AssertionError:rejected.append(name)
            else:raise AssertionError(f'누락/중복 게이트가 놓침: {name}')
    report={'status':'PASS','model_fit_called':False,'model_predict_called':False,'final_lock_scored':False,
            'input_kind':'actual_catboost' if args.actual_catboost else 'fixed_v2_identity_candidate',
            'rows':len(frame),'seeds':common.SEEDS,'validators':common.VALIDATORS,
            'score_cells':len(result['scores'])//2,'bootstrap_blocks':80,'bootstrap_rng_seed':918,
            'identity_zero_delta_ci_and_nonadoption':not bool(args.actual_catboost),'rejected_bad_cases':rejected,
            'independent_math_fsum_scores':independent_scores,'paired_bootstrap_comparison':compared,
            'evaluation':result,'hashes':{'common':sha(COMMON),'catboost':sha(CAT),'audit_code':sha(__file__)}}
    save(args.output_prefix+'.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['evaluation','hashes','independent_math_fsum_scores']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
