"""멤버/원 phase3 캐시와 소스 입력의 보존 해시 검산. 모델 fit 없음."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'
PHASE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert (HERE/'completion.json').exists(),'Wait for original completion'
    manifest=read(HERE/'manifest.json');audits=read(HERE/'reconstruction_audit.json')
    source={'code':sha(HERE/'run_ablation.py')==manifest['code_sha256'],
            'protocol':sha(HERE/'PROTOCOL.md')==manifest['protocol_sha256'],
            'common':sha(ROOT/'집/코덱스/analysis/ec_model_common_20261002_v1/common.py')==manifest['common_sha256'],
            'core':sha(ROOT/'집/코덱스/analysis/codex_independent/rl_ec_v1/run.py')==manifest['core_sha256'],
            'split':sha(ROOT/'집/코덱스/analysis/local/rl_ec_v1/20260927_173801/splits.csv')==manifest['split_sha256'],
            'phase3_manifest':sha(PHASE/'manifest.json')==manifest['phase3_manifest_sha256'],
            'lock':sha(ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json')==manifest['lock_sha256']}
    for name,value in manifest['inputs_sha256'].items():
        source[name]=sha(ROOT/'공용/대회자료/정형데이터/참가자_배포'/name)==value
    assert all(source.values())
    member_ok=[];phase_ok={}
    prefix=json.dumps(manifest,sort_keys=True,ensure_ascii=False)
    for a in audits:
        name=a['validator'];fold=int(a['validation_fold']);seed=int(a['seed'])
        p=OUT/f'{name}_{fold}_seed{seed}_members.npz';metadata=read(p.with_suffix('.json'))
        assert metadata['npz_sha256']==sha(p)
        assert metadata['key'].startswith(prefix)
        assert (metadata['validator'],int(metadata['fold']),int(metadata['seed']))==(name,fold,seed)
        member_ok.append({'validator':name,'fold':fold,'seed':seed,'member_cache_sha256':sha(p)})
        old=PHASE/f'{name}_{fold}.npz';oldmetadata=read(old.with_suffix('.json'))
        assert oldmetadata['prediction_sha256']==sha(old)==a['cache_hash']
        phase_ok[f'{name}/{fold}']=True
    assert len(member_ok)==66 and len(phase_ok)==22
    result={'status':'PASS','source_hash_checks':source,'member_cache_count':len(member_ok),
            'phase3_cache_count':len(phase_ok),'member_cache_checks':member_ok,'phase3_cache_checks':phase_ok,
            'locked_labels_converted':False,'model_fitted':False}
    (HERE/'cache_integrity_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['member_cache_checks','phase3_cache_checks']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
