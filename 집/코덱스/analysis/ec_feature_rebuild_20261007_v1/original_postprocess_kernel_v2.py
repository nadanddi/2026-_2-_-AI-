"""Frozen original recipe arithmetic. No files, labels, models or score gate.

The caller must separately verify all producer receipts and source bindings.
SG2 is provided by the registered reference-only plan; this kernel does not
establish that plan's causality or validate the full model pipeline.
"""
import math


def key(rid):
    if not isinstance(rid, str):
        raise ValueError('row_id must be a string')
    parts = rid.split('_')
    if len(parts) != 3 or parts[0] not in ('F13', 'F47'):
        raise ValueError('unexpected row_id')
    f, d, h = parts[0], int(parts[1]), int(parts[2])
    if d < 0 or not 0 <= h <= 23 or rid != f'{f}_{d:03d}_{h:02d}':
        raise ValueError('noncanonical row_id')
    return f, d, h


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('expected finite numeric scalar')
    return float(value)


def assemble(ids, et, lgb, mlp, pfn_by_context, lower, upper, sg2_apply):
    """Return stage arrays for one baseline/candidate and one model seed.

    Require complete ordered days; PFN contexts are fixed to seeds5..8.
    sg2_apply receives only current-day prediction prefix and RAW_PASS scope.
    Evaluation labels are intentionally absent from this API.
    """
    if not isinstance(ids, (list, tuple)) or not ids or len(set(ids)) != len(ids):
        raise ValueError('nonempty unique ordered IDs required')
    keys = [key(rid) for rid in ids]
    if keys != sorted(keys):
        raise ValueError('IDs must be in canonical farm/day/hour order')
    grouped = {}
    for f, d, h in keys:
        grouped.setdefault((f, d), []).append(h)
    if any(hours != list(range(24)) for hours in grouped.values()):
        raise ValueError('every query day must contain exactly24hours')
    if type(pfn_by_context) is not dict or any(type(s) is not int for s in pfn_by_context) or set(pfn_by_context) != {5, 6, 7, 8}:
        raise ValueError('exact PFN contexts5..8 required')
    def vector(values):
        if not isinstance(values, (list, tuple)) or len(values) != len(ids):
            raise ValueError('prediction length mismatch')
        return [number(v) for v in values]
    et, lgb, mlp = vector(et), vector(lgb), vector(mlp)
    pfn = {s: vector(pfn_by_context[s]) for s in (5, 6, 7, 8)}
    lower, upper = number(lower), number(upper)
    if lower > upper or not callable(sg2_apply):
        raise ValueError('invalid train bounds or SG2 callback')
    def clip(value):
        return min(upper, max(lower, number(value)))
    raw_mix, shrunk, pre, final = [], [], [], []
    day_values, prefix = [], {}
    previous = None
    for i, (rid, (f, d, h)) in enumerate(zip(ids, keys)):
        if previous != (f, d):
            day_values, prefix = [], {}
            previous = (f, d)
        mean_pfn = sum(pfn[s][i] for s in (5, 6, 7, 8)) / 4
        mixed = number(.6 * (.6 * et[i] + .3 * lgb[i] + .1 * mlp[i]) + .4 * mean_pfn)
        raw_mix.append(mixed)
        day_values.append(mixed)
        value = number(.5 * mixed + .5 * math.fsum(day_values) / (h + 1))
        shrunk.append(value)
        pre.append(clip(value))
        prefix[rid] = pre[-1]
        passed = dict(prefix)
        corrected = number(sg2_apply(rid, passed, 'BLK_RAW_PASS'))
        if passed != prefix:
            raise ValueError('SG2 callback mutated its prefix')
        # Registered RAW_PASS policy is identity before original day179.
        if d < 179 and corrected != pre[-1]:
            raise ValueError('SG2 changed a pass1 prediction')
        final.append(clip(corrected))
    return {'raw_mix': raw_mix, 'one_shrink': shrunk,
            'pre_SG2_clip': pre, 'post_SG2_clip': final}

