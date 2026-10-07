from pathlib import Path
import json,hashlib,ast
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((HERE/'CH2_BLK_registration_v3.json').read_text(encoding='utf-8'))
assert reg['statistics']['comparison_alpha']==.025/60
ast.parse((HERE/'score_ch2_BLK_v2.py').read_text(encoding='utf-8'))
ast.parse((HERE/'verify_ch2_BLK_v3.py').read_text(encoding='utf-8'))
spec={'scorer_sha256':sha(HERE/'score_ch2_BLK_v2.py'),'verifier_sha256':sha(HERE/'verify_ch2_BLK_v3.py'),
      'registration_sha256':sha(HERE/'CH2_BLK_registration_v3.json'),'cumulative_comparisons':60,
      'alpha':.025/60,'draws':20000,'rng_seed':2026100702,'segments':32,
      'loss':'mean_seed squared-error delta','bootstrap':'farm-stratified4blocks per farm row-weighted',
      'p':'plusone/ties>=0','individual_CI':'95% descriptive','adjusted_CI':'1-.05/60',
      'adoption_permitted':False,'created_before_inference':False,'repair_scope':'JSON tuple/list equivalence only; original statistics/method/predictions unchanged; before truth score'}
out=HERE/'CH2_BLK_score_registration_v2.json';assert not out.exists()
out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
print('CH2 scorer/statistics/32segment registration saved before inference')
