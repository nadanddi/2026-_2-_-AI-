"""Post-hoc concentration diagnostics on frozen confirmation predictions."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change(base_sq, candidate_sq):
    return float(np.sqrt(candidate_sq / base_sq) - 1)


def main():
    out = Path(sys.argv[1]).resolve()
    assert out.is_dir() and ROOT / 'local/ec_locked_confirmation' in out.parents
    assert (out / 'result.json').is_file()
    files = [out / f'fold{fold}.csv' for fold in (8, 9)]
    a = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    assert a.row_id.is_unique
    a['base_sq'] = (a.sub_ec - a.baseline) ** 2
    a['candidate_sq'] = (a.sub_ec - a.candidate) ** 2
    a['gain_sq'] = a.base_sq - a.candidate_sq
    total_base = float(a.base_sq.sum())
    total_candidate = float(a.candidate_sq.sum())
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert abs(change(total_base, total_candidate) - result['pooled']['relative_change']) < 1e-12

    blocks = a.groupby(['farm', 'block'])[['base_sq', 'candidate_sq', 'gain_sq']].sum().reset_index()
    assert len(blocks) == 16 and set(blocks.farm) == {'F13', 'F47'}
    leave_block = []
    for r in blocks.itertuples():
        leave_block.append(dict(farm=r.farm, block=int(r.block),
            relative_change=change(total_base-r.base_sq, total_candidate-r.candidate_sq),
            removed_gain_sq=float(r.gain_sq)))

    days = a.groupby(['farm', 'day'])[['base_sq', 'candidate_sq', 'gain_sq']].sum().reset_index()
    days = days.sort_values('gain_sq', ascending=False)
    remove_top_days = {}
    for k in (1, 2, 3, 5):
        top = days.head(k)
        remove_top_days[str(k)] = change(total_base-float(top.base_sq.sum()),
                                         total_candidate-float(top.candidate_sq.sum()))

    gains = blocks.gain_sq.to_numpy(float)
    bits = ((np.arange(1 << len(gains), dtype=np.uint32)[:, None] >>
             np.arange(len(gains), dtype=np.uint32)) & 1).astype(bool)
    sign_flipped = np.where(bits, gains, -gains).sum(axis=1)
    observed = float(gains.sum())
    signflip_p_one_sided = float(np.mean(sign_flipped >= observed - 1e-12))
    output = dict(status='POST_HOC_STRESS_TEST',
        code_hash=sha(HERE), result_hash=sha(out / 'result.json'),
        fold_files={file.name: sha(file) for file in files},
        pooled_relative_change=change(total_base, total_candidate),
        positive_blocks=int((blocks.gain_sq > 0).sum()), total_blocks=len(blocks),
        leave_one_block_out=dict(min_relative_change=float(min(x['relative_change'] for x in leave_block)),
                                 max_relative_change=float(max(x['relative_change'] for x in leave_block)),
                                 all_improve=bool(all(x['relative_change'] < 0 for x in leave_block)),
                                 details=leave_block),
        remove_top_improving_days=remove_top_days,
        block_sign_flip=dict(one_sided_p=signflip_p_one_sided,
                             permutations=len(sign_flipped),
                             assumption='Independent symmetric block gains under a null of zero effect; not adjusted for candidate search.'),
        note='Descriptive stress test after seeing confirmation; no model selection or new independent validation.')
    (out / 'robustness.json').write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
