"""Synthetic Decimal-vs-float replay and refusal tests; no competition data."""
from pathlib import Path
import copy
import hashlib
import json
import math
import original_independent_arithmetic_v1 as independent
import original_postprocess_kernel_v2 as production

HERE = Path(__file__).resolve().parent


def main():
    source_names = ('audit_original_independent_arithmetic_v2.py',
                    'original_independent_arithmetic_v1.py', 'original_postprocess_kernel_v2.py')
    source_sha = {str((HERE/n).resolve()): hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in source_names}
    ids = [f'{f}_{d:03d}_{h:02d}' for f in ('F13', 'F47') for d in (178, 179) for h in range(24)]
    et = [0.05 if h < 3 else (3.2 if h >= 18 else .8 + h*.003) for _, _, h in map(independent.row_key, ids)]
    lgb = [.9]*len(ids); mlp = [1.3]*len(ids)
    pfns = {s: [.7 + s*.01 + h*.002 for _, _, h in map(independent.row_key, ids)] for s in (5, 6, 7, 8)}
    choices = {}
    for rid in ids:
        f, d, h = independent.row_key(rid)
        selected = d >= 179 and h % 3 != 0
        choices[rid] = {'active': d >= 179, 'has_candidate': selected,
                        'reference_day': 170 if selected else None,
                        'level': (1.1 if h < 18 else 2.5) if selected else None}
    def callback(rid, prefix, scope):
        assert scope == 'BLK_RAW_PASS'
        c = choices[rid]; current = prefix[rid]
        if not c['has_candidate']: return current
        delta = c['level'] - math.fsum(prefix.values())/len(prefix)
        return current + delta/2 if abs(delta) <= .30 else current
    saved = production.assemble(ids, et, lgb, mlp, pfns, .3, 1.5, callback)
    train_days = {('F13', 170), ('F47', 170)}
    result = independent.verify(ids, et, lgb, mlp, pfns, [.3, 1.5], choices, saved, train_days)
    assert result['rows'] == 96 and sum(result['stage_comparisons'].values()) == 384
    refused = []
    def reject(name, args):
        try: independent.verify(**args)
        except ValueError: refused.append(name)
        else: raise AssertionError('invalid accepted: '+name)
    args = dict(ids=ids, et=et, lgb=lgb, mlp=mlp, pfns=pfns, bounds=[.3, 1.5],
                choices=choices, saved=saved, train_days=train_days)
    for stage in independent.STAGES:
        broken = copy.deepcopy(saved); broken[stage][30] += .0001
        reject('tampered '+stage, dict(args, saved=broken))
    reject('reversed bounds', dict(args, bounds=[1.5, .3]))
    reject('PFN context omission', dict(args, pfns={5:pfns[5],6:pfns[6],7:pfns[7]}))
    reject('incomplete query day', dict(args, ids=ids[:-1]))
    reject('nonfinite ET', dict(args, et=[float('nan')]+et[1:]))
    reject('boolean ET', dict(args, et=[True]+et[1:]))
    reject('foreign train reference', dict(args, train_days={('F13', 170)}))
    wrong = copy.deepcopy(choices); wrong[ids[0]]['active'] = True
    reject('pass1 activation', dict(args, choices=wrong))
    wrong = copy.deepcopy(choices); wrong[ids[25]]['reference_day'] = 179
    reject('current query day as reference', dict(args, choices=wrong))
    wrong = copy.deepcopy(choices); wrong[ids[25]]['level'] = math.fsum(saved['pre_SG2_clip'][24:26])/2+.30
    reject('SG2 threshold ambiguity', dict(args, choices=wrong))
    wrong = copy.deepcopy(choices); wrong[ids[0]]['level'] = 1.1
    reject('inactive choice with level', dict(args, choices=wrong))
    # Values of later predictions cannot alter the earlier reconstructed outputs.
    poison_checks = 0
    for index in (6, 30, 54, 78):
        farm, day, hour = independent.row_key(ids[index])
        poisoned = list(et)
        for j, rid in enumerate(ids):
            f, d, h = independent.row_key(rid)
            if f != farm or d > day or (d == day and h > hour): poisoned[j] = 999.
        later = production.assemble(ids, poisoned, lgb, mlp, pfns, .3, 1.5, callback)
        for stage in independent.STAGES: assert saved[stage][index] == later[stage][index]
        independent.verify(ids, poisoned, lgb, mlp, pfns, [.3, 1.5], choices, later, train_days)
        poison_checks += 1
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path,value in source_sha.items())
    audit = {'status': 'INDEPENDENT_RECIPE_SYNTHETIC_PASS_NOT_FULL_GATE',
             'decimal_precision': 60, 'decimal_stage_comparisons': 384,
             'conditional_replay': result, 'invalid_checks': refused, 'invalid_count': len(refused),
             'prediction_prefix_poison_cases': poison_checks, 'executed_sources_sha256': source_sha,
             'model_fit': False, 'heldout_truth_read': False, 'whole_pipeline_gate_passed': False,
             'limits': ['Prediction-prefix arithmetic only; no query-input causal claim',
                        'SG2 choices and bounds supplied; no independent reference-selection proof',
                        'Actual original66 predictions not loaded or verified here']}
    with (HERE/'original_independent_arithmetic_synthetic_audit_v2.json').open('x', encoding='utf-8') as out:
        json.dump(audit, out, ensure_ascii=False, indent=2)
    print('Independent Decimal recipe synthetic PASS',384,'comparisons;',len(refused),'refusals;',poison_checks,'prefix cases')


if __name__ == '__main__': main()
