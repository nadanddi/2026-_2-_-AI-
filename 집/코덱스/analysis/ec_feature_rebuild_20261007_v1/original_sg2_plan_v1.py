"""Original RAW_PASS SG2 choices, selected lazily with recorded source bindings.

This adapter is not a whole-model or scoring gate. Its caller must separately
prove reference context membership, matrix lineage and query-input causality.
"""
import math
from pathlib import Path
from types import MappingProxyType
import blk_sg2_refonly_v1 as sg1
import blk_sg2_refonly_v2 as sg2
from blk_baseline_data_v1 import HERE, sha, np
from checkpoint_v1 import digest
from original_postprocess_kernel_v2 import key, number


class OriginalSG2Plan:
    def __init__(self, ctx, ids, registered_sources):
        self.ids = tuple(ids)
        if len(set(self.ids)) != len(self.ids) or not self.ids:
            raise ValueError('nonempty unique query IDs required')
        for rid in self.ids: key(rid)
        if set(self.ids) != set(ctx.query_ids):
            raise ValueError('query membership mismatch')
        required = [Path(__file__).resolve(), Path(sg1.__file__).resolve(),
                    Path(sg2.__file__).resolve(), sg1.SG2_SOURCE.resolve(),
                    HERE / 'original_postprocess_kernel_v2.py']
        if Path(sg1.__file__).resolve() != (HERE / 'blk_sg2_refonly_v1.py').resolve():
            raise ValueError('wrong SG2 base module')
        if Path(sg2.__file__).resolve() != (HERE / 'blk_sg2_refonly_v2.py').resolve():
            raise ValueError('wrong SG2 prefix adapter')
        self.sources = {str(p): sha(p) for p in required}
        if any(registered_sources.get(p) != value for p, value in self.sources.items()):
            raise ValueError('unregistered or changed SG2 sources')
        self.sg = sg2.RefOnlySG2(ctx)
        self._choices = {}
        self._source_checks = 0
        self._maximum = 0.

    def choice(self, rid):
        if rid not in self.ids:
            raise ValueError('unknown query row')
        if rid not in self._choices:
            f, d, h = key(rid)
            if d < 179:
                choice = {'active': False, 'has_candidate': False,
                          'reference_day': None, 'level': None}
            else:
                zero = {f'{f}_{d:03d}_{j:02d}': 0. for j in range(h + 1)}
                _, diag = self.sg.predict_one(rid, zero, 'BLK_RAW_PASS')
                if type(diag['active']) is not bool or type(diag['has_candidate']) is not bool:
                    raise ValueError('invalid source flags')
                chosen = diag.get('reference_day')
                if diag['has_candidate']:
                    if (f, chosen) not in self.sg.ref or (f, chosen) == (f, d):
                        raise ValueError('SG2 selection is not a distinct training reference')
                    level = number(float(self.sg.ec[f, chosen]))
                    chosen = int(chosen)
                else:
                    chosen, level = None, None
                choice = {'active': diag['active'], 'has_candidate': diag['has_candidate'],
                          'reference_day': chosen, 'level': level}
            if not choice['active'] and choice['has_candidate']:
                raise ValueError('inactive SG2 cannot select a reference')
            self._choices[rid] = MappingProxyType(choice)
        return self._choices[rid]

    def apply(self, rid, prefix, scope):
        if scope != 'BLK_RAW_PASS':
            raise ValueError('original pipeline only uses RAW_PASS')
        f, d, h = key(rid)
        expected = [f'{f}_{d:03d}_{j:02d}' for j in range(h + 1)]
        if type(prefix) is not dict or list(prefix) != expected:
            raise ValueError('ordered complete prediction prefix required')
        values = [number(prefix[r]) for r in expected]
        chosen = self.choice(rid)
        current = values[-1]
        if not chosen['has_candidate']:
            return current
        difference = chosen['level'] - float(np.mean(values))
        return current + .5 * difference if abs(difference) <= .30 else current

    def audit_source(self, rid, prefix):
        planned = self.apply(rid, prefix, 'BLK_RAW_PASS')
        source, diag = self.sg.predict_one(rid, dict(prefix), 'BLK_RAW_PASS')
        chosen = self.choice(rid)
        if diag['active'] != chosen['active'] or diag['has_candidate'] != chosen['has_candidate']:
            raise ValueError('source selection changed')
        if chosen['has_candidate'] and diag['reference_day'] != chosen['reference_day']:
            raise ValueError('source reference day changed')
        error = abs(number(float(source)) - planned)
        if not math.isfinite(error) or error > 1e-12:
            raise ValueError('SG2 source arithmetic mismatch')
        self._source_checks += 1
        self._maximum = max(self._maximum, error)
        return error

    def snapshot(self, require_complete=True):
        if require_complete and set(self._choices) != set(self.ids):
            raise ValueError('SG2 choices are incomplete')
        if any(sha(p) != value for p, value in self.sources.items()):
            raise ValueError('SG2 source changed during use')
        choices = {rid: dict(self._choices[rid]) for rid in self.ids if rid in self._choices}
        return {'status': 'ORIGINAL_SG2_CHOICES_NOT_FULL_MODEL_GATE',
                'row_ids': list(self.ids), 'choices': choices, 'choices_sha256': digest(choices),
                'sources_sha256': dict(self.sources), 'scope': 'BLK_RAW_PASS',
                'source_checks': self._source_checks, 'source_maximum_difference': self._maximum,
                'all_choices_present': set(choices) == set(self.ids),
                'heldout_truth_loaded': False, 'whole_pipeline_gate_passed': False}
