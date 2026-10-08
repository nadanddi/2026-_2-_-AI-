"""One-fold actual SG2 source audit with synthetic prediction prefixes; no scores."""
from pathlib import Path
import json
import sys
from original_sg2_plan_v1 import OriginalSG2Plan
from blk_context_v1 import BLKContext, key
from blk_baseline_data_v1 import HERE, sha


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    registration = HERE / 'ORIGINAL_SG2_source_probe_registration_v1.json'
    reg = json.loads(registration.read_text(encoding='utf-8'))
    assert reg['status'] == 'REGISTERED_ONE_FOLD_SG2_SOURCE_AUDIT_NO_SCORE'
    assert reg['probe_sha256'] == sha(__file__)
    assert all(sha(p) == v for p, v in reg['sources_sha256'].items())
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    fold = next(f for f in registry['folds'] if f['validator'] == 'DIAG10' and f['fold'] == 0)
    ids = fold['ordered_query_ids']
    ctx = BLKContext({'status': 'REGISTERED', 'train_ids': fold['ordered_train_ids'],
                      'query_ids': ids, 'gap_ids_REMOVE_INPUT_AND_BOTH_LABELS': fold['input_forbidden_ids']})
    assert set(ctx.reference_inputs) == set(ctx.reference_labels) == set(fold['ordered_train_ids'])
    assert not set(ctx.reference_labels).intersection(ctx.query_ids | ctx.gap_ids)
    plan = OriginalSG2Plan(ctx, ids, reg['sources_sha256'])
    def prefix(rid):
        f, d, h = key(rid)
        # Fixed synthetic values test adapter/source arithmetic without revealing
        # original heldout EC or selecting a performance-dependent recipe.
        return {f'{f}_{d:03d}_{j:02d}': .8 + .001 * j for j in range(h + 1)}
    checked = []
    maximum = 0.
    for rid in ids:
        maximum = max(maximum, plan.audit_source(rid, prefix(rid)))
        checked.append(rid)
    assert checked == ids
    snapshot = plan.snapshot()
    assert snapshot['source_checks'] == len(ids)
    probes = []
    for farm in ('F13', 'F47'):
        days = sorted({key(r)[1] for r in ids if key(r)[0] == farm})
        for phase in (False, True):
            options = [d for d in days if (d >= 179) == phase]
            if options:
                probes += [f'{farm}_{options[0]:03d}_{h:02d}' for h in (0, 6, 23)]
    poison_audits = []
    for rid in probes:
        f, d, h = key(rid)
        packet = ctx.query_prefix(rid)
        permitted = set(packet)
        forbidden = set(ctx.query_ids) - permitted
        assert all(key(r)[0] == f and key(r)[1:] <= (d, h) for r in packet)
        previous = {r: ctx._query[r] for r in forbidden}
        try:
            for r in forbidden: ctx._query[r] = {c: 77777. for c in previous[r]}
            assert ctx.query_prefix(rid) == packet
            # Fresh instance and empty cache force actual choice reconstruction.
            fresh = OriginalSG2Plan(ctx, ids, reg['sources_sha256'])
            assert not fresh._choices
            expected = plan.apply(rid, prefix(rid), 'BLK_RAW_PASS')
            actual = fresh.apply(rid, prefix(rid), 'BLK_RAW_PASS')
            assert dict(fresh.choice(rid)) == dict(plan.choice(rid))
            assert abs(actual - expected) <= 1e-12
            fresh.audit_source(rid, prefix(rid))
            partial = fresh.snapshot(require_complete=False)
            assert list(partial['choices']) == [rid]
            poison_audits.append({'row_id': rid, 'permitted_query_rows': len(permitted),
                                  'forbidden_query_rows': len(forbidden),
                                  'choice_equal': True, 'difference': abs(actual - expected),
                                  'fresh_cache': True})
        finally:
            ctx._query.update(previous)
        print(f'SG2 fresh source/future audit {rid} PASS', flush=True)
    assert all(sha(p) == v for p, v in reg['sources_sha256'].items())
    result = {'status': 'ONE_FOLD_ACTUAL_SG2_SOURCE_PASS_NOT_FULL_MODEL_GATE',
              'registration_sha256': sha(registration), 'probe_sha256': sha(__file__),
              'validator': 'DIAG10', 'fold': 0, 'row_ids_checked': checked,
              'source_checks': len(checked), 'maximum_difference': maximum,
              'choice_snapshot': snapshot, 'fresh_future_poison_audits': poison_audits,
              'heldout_truth_loaded': False, 'prediction_prefixes_synthetic': True,
              'model_fit': False, 'whole_pipeline_gate_passed': False,
              'limits': ['One fold only; no baseline/candidate mixed prediction verification',
                         'Fresh poison probes do not constitute all-row universal causality proof',
                         'Original producer/PFN receipts and full scoring gate remain mandatory']}
    with (HERE / 'ORIGINAL_SG2_source_probe_result_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'Actual SG2 source one fold PASS: {len(checked)} rows, {len(probes)} fresh poison probes; no score')


if __name__ == '__main__': main()
