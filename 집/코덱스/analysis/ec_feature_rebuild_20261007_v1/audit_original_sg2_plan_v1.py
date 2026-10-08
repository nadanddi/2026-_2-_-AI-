"""Synthetic adapter contract tests, not actual SG2 reference selection proof."""
from pathlib import Path
from types import SimpleNamespace
import json
import original_sg2_plan_v1 as module

HERE = Path(__file__).resolve().parent


def main():
    ids = ['F13_178_00', 'F13_179_00', 'F13_179_01', 'F13_179_23', 'F47_179_00']
    required = [Path(module.__file__).resolve(), Path(module.sg1.__file__).resolve(),
                Path(module.sg2.__file__).resolve(), module.sg1.SG2_SOURCE.resolve(),
                HERE / 'original_postprocess_kernel_v2.py']
    bindings = {str(p): module.sha(p) for p in required}
    ctx = SimpleNamespace(query_ids=set(ids))
    class Oracle:
        def __init__(self, ctx):
            self.ref = {('F13', 170), ('F47', 170)}
            self.ec = {r: 1. for r in self.ref}
            self.calls = []; self.wrong = False
        def predict_one(self, rid, prefix, scope):
            self.calls.append(rid)
            f, d, h = module.key(rid)
            if d < 179:
                return prefix[rid], {'active': False, 'has_candidate': False}
            delta = 1. - float(module.np.mean(list(prefix.values())))
            v = prefix[rid] + .5 * delta if abs(delta) <= .30 else prefix[rid]
            return v + (1. if self.wrong else 0.), {'active': True, 'has_candidate': True,
                                                      'reference_day': 170}
    original = module.sg2.RefOnlySG2
    module.sg2.RefOnlySG2 = Oracle
    checks = 0
    def reject(fn):
        nonlocal checks
        try: fn()
        except (ValueError, KeyError): checks += 1
        else: raise AssertionError('invalid adapter contract accepted')
    try:
        plan = module.OriginalSG2Plan(ctx, ids, bindings)
        assert plan._choices == {} and plan.sg.calls == []
        assert plan.apply(ids[0], {ids[0]: .8}, 'BLK_RAW_PASS') == .8
        assert plan.sg.calls == []  # Original pass1 does not even select reference.
        reject(plan.snapshot)
        maxima = 0.
        for rid in ids:
            f, d, h = module.key(rid)
            prefix = {f'{f}_{d:03d}_{j:02d}': .8 for j in range(h + 1)}
            before = dict(prefix)
            v = plan.apply(rid, prefix, 'BLK_RAW_PASS')
            assert abs(v - (.8 if d < 179 else .9)) < 1e-12 and prefix == before
            maxima = max(maxima, plan.audit_source(rid, prefix))
        cached = len(plan.sg.calls)
        for _ in range(3): plan.apply(ids[1], {ids[1]: .8}, 'BLK_RAW_PASS')
        assert len(plan.sg.calls) == cached
        snap = plan.snapshot()
        assert snap['all_choices_present'] and snap['source_checks'] == len(ids)
        assert snap['choices'][ids[0]]['reference_day'] is None
        assert snap['choices'][ids[-1]]['reference_day'] == 170
        assert plan.apply(ids[1], {ids[1]: .1}, 'BLK_RAW_PASS') == .1  # Guard rejects large level difference.
        reject(lambda: plan.apply(ids[1], {ids[1]: .8}, 'BLK_QUERY_ROLE'))
        reject(lambda: plan.apply(ids[2], {ids[2]: .8}, 'BLK_RAW_PASS'))
        reject(lambda: plan.apply(ids[2], {ids[2]: .8, ids[1]: .8}, 'BLK_RAW_PASS'))
        reject(lambda: plan.apply(ids[1], {ids[1]: float('nan')}, 'BLK_RAW_PASS'))
        reject(lambda: plan.apply(ids[1], {ids[1]: True}, 'BLK_RAW_PASS'))
        reject(lambda: plan.choice('F13_179_02'))
        reject(lambda: module.OriginalSG2Plan(ctx, ids + [ids[0]], bindings))
        reject(lambda: module.OriginalSG2Plan(ctx, ids[:-1], bindings))
        reject(lambda: module.OriginalSG2Plan(ctx, ids, {}))
        plan.sg.wrong = True
        reject(lambda: plan.audit_source(ids[1], {ids[1]: .8}))
        plan.sg.wrong = False
        plan.sources[next(iter(plan.sources))] = '0' * 64
        reject(plan.snapshot)
        result = {'status': 'SYNTHETIC_ORIGINAL_SG2_ADAPTER_PASS_NOT_ACTUAL_SG2_OR_FULL_GATE',
                  'source_scalar_checks': len(ids), 'invalid_contracts_rejected': checks,
                  'maximum_source_difference': maxima, 'lazy_cache_reuse': True,
                  'pass1_selection_skipped': True, 'actual_SG2_constructor_called': False,
                  'sources_sha256': {str(p): module.sha(p) for p in required + [Path(__file__)]},
                  'heldout_truth_read': False, 'model_fit': False, 'whole_pipeline_gate_passed': False,
                  'limits': ['Oracle is synthetic; actual reference selection and context source boundaries not tested',
                             'Original66 producer receipts and real all-row SG2/future-input checks remain mandatory']}
        with (HERE / 'original_sg2_plan_synthetic_audit_v1.json').open('x', encoding='utf-8') as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
        print(f'Synthetic SG2 adapter PASS: {len(ids)} source scalars, {checks} rejections; no actual SG2 fit')
    finally:
        module.sg2.RefOnlySG2 = original


if __name__ == '__main__': main()
