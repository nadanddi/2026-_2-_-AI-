"""Resume only unfinished pre-registered K2/K3 after EC delivery verification."""
from pathlib import Path
import sys,os,json,subprocess
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
K2=HERE.parent/'ec_stage2_tabpfn_20261002_v2'
LOCAL=ROOT/'집/코덱스/local/ec_stage2_tabpfn_20261002_v2'
DC4=HERE.parent/'ec_dc4_integration_20261002_v1'
DELIVERY=ROOT/'집/코덱스/local/ec_submission10_season_20261002_v2/artifact'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def call(script,*args):
    print('Resume '+script.name+' '+' '.join(args),flush=True)
    subprocess.run([sys.executable,'-B','-u',str(script),*args],env=os.environ.copy(),check=True)
def main():
    os.environ['PYTHONPATH']='';os.environ['PYTHONIOENCODING']='utf-8'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
    assert read(DELIVERY/'delivery_manifest.json')['status']=='PASS'
    assert read(DC4/'DC4_판정_v1.json')['candidate_adopted']
    assert read(DC4/'locked_confirmation_result.json')['status']=='CONSUMED'
    assert not list(LOCAL.glob('*.lock')),'Existing worker must finish before starting another'
    for shard in range(3):assert read(LOCAL/f'shard{shard}_r3_completion.json')['status']=='COMPLETE'
    for shard in range(3):
        marker=LOCAL/f'shard{shard}_pfn_completion.json'
        if not marker.exists():call(K2/'run_stage2_tabpfn.py','--run','--component','pfn','--shard',str(shard))
        else:assert read(marker)['status']=='COMPLETE'
    if not (K2/'result.json').exists():call(K2/'run_stage2_tabpfn.py','--collect')
    if not (K2/'independent_verification_v2.json').exists():call(K2/'verify_stage2_v2.py')
    assert read(K2/'independent_verification_v2.json')['status']=='PASS'
    target=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/queue_complete.json'
    if not target.exists():
        record={'status':'COMPLETE','order_version':4,'stages':['DC4_ET','DC4_v2','locked_once','submission10_EC_blank_temp','K2_K3'],'platform_submission':False,'K2_candidate_adopted':False,'final_review_pending':True}
        with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print('K2/K3 fixed calculation and independent audit completed; report helper can finish.',flush=True)
if __name__=='__main__':main()
