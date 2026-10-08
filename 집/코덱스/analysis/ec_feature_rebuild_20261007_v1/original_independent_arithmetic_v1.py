"""Decimal60 replay of saved recipe stages, independent of production kernels.

This validates arithmetic conditional on supplied predictions/bounds/SG2 choices.
It does not establish model, feature, training-label or SG2 selection causality.
"""
from decimal import Decimal, localcontext
import math

STAGES = ('raw_mix', 'one_shrink', 'pre_SG2_clip', 'post_SG2_clip')


def num(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite scalar required')
    return Decimal(str(value))


def row_key(rid):
    if type(rid) is not str: raise ValueError('row ID type')
    pieces = rid.split('_')
    if len(pieces) != 3 or pieces[0] not in ('F13', 'F47'): raise ValueError('row ID')
    farm, day, hour = pieces[0], int(pieces[1]), int(pieces[2])
    if day < 0 or not 0 <= hour < 24 or rid != f'{farm}_{day:03d}_{hour:02d}':
        raise ValueError('noncanonical row ID')
    return farm, day, hour


def verify(ids, et, lgb, mlp, pfns, bounds, choices, saved, train_days, tolerance=1e-12):
    if tolerance != 1e-12: raise ValueError('fixed tolerance required')
    if type(ids) is not list or not ids or len(ids) != len(set(ids)):
        raise ValueError('unique nonempty ordered IDs required')
    keys = [row_key(rid) for rid in ids]
    if keys != sorted(keys): raise ValueError('row order')
    groups = {}
    for f, d, h in keys: groups.setdefault((f, d), []).append(h)
    if any(h != list(range(24)) for h in groups.values()): raise ValueError('complete day required')
    if type(pfns) is not dict or any(type(s) is not int for s in pfns) or set(pfns) != {5, 6, 7, 8}:
        raise ValueError('PFN context set')
    if type(saved) is not dict or set(saved) != set(STAGES): raise ValueError('stage set')
    if type(choices) is not dict or set(choices) != set(ids): raise ValueError('choice coverage')
    if type(train_days) is not set or any(type(t) is not tuple or len(t) != 2 for t in train_days):
        raise ValueError('training day membership set')
    if type(bounds) is not list or len(bounds) != 2: raise ValueError('bounds')
    lo, hi = map(num, bounds)
    if lo > hi: raise ValueError('reversed bounds')
    def vector(v):
        if type(v) is not list or len(v) != len(ids): raise ValueError('vector shape')
        return [num(x) for x in v]
    e, l, m = vector(et), vector(lgb), vector(mlp)
    p = {s: vector(v) for s, v in pfns.items()}
    stages = {name: vector(v) for name, v in saved.items()}
    maximum = {name: Decimal(0) for name in STAGES}
    count = {name: 0 for name in STAGES}
    prefix_mix, prefix_clipped, previous = [], [], None
    def clip(v): return max(lo, min(hi, v))
    with localcontext() as ctx:
        ctx.prec = 60
        for i, (rid, (f, d, h)) in enumerate(zip(ids, keys)):
            if previous != (f, d): prefix_mix, prefix_clipped, previous = [], [], (f, d)
            mixed = Decimal('.6')*(Decimal('.6')*e[i]+Decimal('.3')*l[i]+Decimal('.1')*m[i])
            mixed += Decimal('.4')*sum(p[s][i] for s in (5, 6, 7, 8))/4
            prefix_mix.append(mixed)
            shrunk = (mixed + sum(prefix_mix)/(h+1))/2
            pre = clip(shrunk); prefix_clipped.append(pre)
            choice = choices[rid]
            if type(choice) is not dict or set(choice) != {'active', 'has_candidate', 'reference_day', 'level'}:
                raise ValueError('choice schema')
            if type(choice['active']) is not bool or type(choice['has_candidate']) is not bool:
                raise ValueError('choice flags')
            if choice['active'] != (d >= 179): raise ValueError('RAW_PASS original-day activation')
            post = pre
            if choice['has_candidate']:
                ref = choice['reference_day']
                if not choice['active'] or type(ref) is not int or ref == d or (f, ref) not in train_days:
                    raise ValueError('distinct samefarm training reference required')
                delta = num(choice['level']) - sum(prefix_clipped)/(h+1)
                if abs(abs(delta) - Decimal('.30')) <= Decimal('1e-12'):
                    raise ValueError('SG2 threshold numerically ambiguous; independent source replay required')
                if abs(delta) < Decimal('.30'): post = pre + delta/2
            elif choice['reference_day'] is not None or choice['level'] is not None:
                raise ValueError('unselected choice has reference values')
            expected = dict(zip(STAGES, (mixed, shrunk, pre, clip(post))))
            for name in STAGES:
                error = abs(expected[name] - stages[name][i])
                if error > Decimal('1e-12'): raise ValueError('recipe arithmetic mismatch: '+rid+'/'+name)
                maximum[name] = max(maximum[name], error); count[name] += 1
    return {'rows': len(ids), 'stage_comparisons': count,
            'maximum_difference': {k: float(v) for k, v in maximum.items()},
            'status': 'CONDITIONAL_DECIMAL_ARITHMETIC_PASS_NOT_FULL_PIPELINE_GATE',
            'whole_pipeline_gate_passed': False}
