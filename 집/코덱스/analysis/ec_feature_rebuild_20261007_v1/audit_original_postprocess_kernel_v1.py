"""Synthetic independent Decimal arithmetic and prefix boundary tests only."""
from pathlib import Path
import copy
import hashlib
import json
from decimal import Decimal, localcontext
from original_postprocess_kernel_v1 import assemble

HERE = Path(__file__).resolve().parent


def main():
    ids = [f'{f}_{d:03d}_{h:02d}' for f in ('F13', 'F47') for d in (178, 179) for h in range(24)]
    n = len(ids)
    et = [(i % 29) / 17 for i in range(n)]
    lgb = [(i % 19) / 13 for i in range(n)]
    mlp = [(i % 23) / 11 for i in range(n)]
    pf = {s: [(i % 31 + s) / 21 for i in range(n)] for s in (5, 6, 7, 8)}
    bounds = (.2, 1.4)
    calls = []
    def sg(rid, prefix, scope):
        f, d, h = rid.split('_'); d, h = int(d), int(h)
        assert scope == 'BLK_RAW_PASS'
        assert list(prefix) == [f'{f}_{d:03d}_{j:02d}' for j in range(h + 1)]
        calls.append(rid)
        return prefix[rid] if d < 179 else prefix[rid] + .08
    result = assemble(ids, et, lgb, mlp, pf, *bounds, sg)
    assert len(calls) == n
    maximum = 0.
    with localcontext() as ctx:
        ctx.prec = 60
        D = lambda v: Decimal(str(v))
        mixtures = []
        for i in range(n):
            mixed = D('.6') * (D('.6') * D(et[i]) + D('.3') * D(lgb[i]) + D('.1') * D(mlp[i]))
            mixed += D('.4') * sum(D(pf[s][i]) for s in (5, 6, 7, 8)) / D(4)
            mixtures.append(mixed)
            h = i % 24
            shrunk = (mixed + sum(mixtures[i-h:i+1]) / D(h + 1)) / D(2)
            pre = min(D(bounds[1]), max(D(bounds[0]), shrunk))
            day = int(ids[i].split('_')[1])
            final = min(D(bounds[1]), max(D(bounds[0]), pre + (D('.08') if day >= 179 else D(0))))
            for name, value in zip(result, (mixed, shrunk, pre, final)):
                error = abs(float(value) - result[name][i])
                assert error < 1e-12
                maximum = max(maximum, error)
    # Poison all later hours/days and the other farm for each selected boundary.
    probes = [0, 6, 23, 24, 30, 47, 48, 71, 72, 95]
    for i in probes:
        f, d, h = ids[i].split('_'); d, h = int(d), int(h)
        poisoned = [list(et), list(lgb), list(mlp)]
        pp = copy.deepcopy(pf)
        for j, rid in enumerate(ids):
            g, e, k = rid.split('_'); e, k = int(e), int(k)
            if g != f or (e, k) > (d, h):
                for values in poisoned: values[j] = 9000.
                for values in pp.values(): values[j] = -9000.
        replay = assemble(ids, *poisoned, pp, *bounds, sg)
        assert all(replay[name][i] == result[name][i] for name in result)
    base = [ids, et, lgb, mlp, pf, *bounds, sg]
    bad = []
    def case(index, value):
        args = copy.deepcopy(base[:7]) + [sg]
        args[index] = value; bad.append(args)
    case(0, ids[:-1]); case(0, [ids[0]] + ids[:-1]); case(0, list(reversed(ids)))
    case(0, ['F13_178_24'] + ids[1:]); case(0, ['F13_0178_00'] + ids[1:])
    case(1, et[:-1]); case(1, [float('nan')] + et[1:]); case(2, [True] + lgb[1:])
    case(3, [float('inf')] + mlp[1:]); case(4, {s: pf[s] for s in (5, 6, 7)})
    case(4, {**pf, 9: et}); case(5, 2.); case(6, float('nan'))
    case(7, lambda rid, p, scope: p[rid] + 1.)
    def mutation(rid, p, scope):
        v = p[rid]; p.clear(); return v
    case(7, mutation)
    for args in bad:
        try: assemble(*args)
        except (ValueError, AssertionError): pass
        else: raise AssertionError('invalid input accepted')
    payload = {'status': 'SYNTHETIC_KERNEL_PASS_NOT_FULL_PIPELINE_GATE',
        'rows': n, 'decimal_stage_checks': n * 4, 'maximum_difference': maximum,
        'future_other_farm_poison_probes': len(probes), 'invalid_inputs_rejected': len(bad),
        'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
             (Path(__file__), HERE / 'original_postprocess_kernel_v1.py')},
        'model_fit': False, 'heldout_truth_read': False, 'full_pipeline_gate_passed': False,
        'limits': ['Synthetic callback only; no actual SG2 choice reconstruction',
                   'Producer model/matrix/receipt validation and scoring registration remain mandatory']}
    with (HERE / 'original_postprocess_kernel_synthetic_audit_v1.json').open('x', encoding='utf-8') as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
    print(f'Synthetic kernel PASS: {n*4} Decimal stages, {len(probes)} poison, {len(bad)} rejections')


if __name__ == '__main__': main()
