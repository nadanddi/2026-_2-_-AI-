"""Predeclared humidity-change proxy versus EC drop events on public labels."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parents[0] / '온라인대회자료/정형데이터/참가자_배포'
COLS = ['row_id', 'in_hum', 'in_temp', 'out_hum', 'act_fog', 'act_vent']


def summaries(frame, proxy, outcome):
    result = {}
    for farm, group in [('ALL', frame), *list(frame.groupby('farm'))]:
        p = proxy[group.index]
        y = outcome[group.index]
        n1, n0 = int(p.sum()), int((~p).sum())
        a, b = int((p & y).sum()), int((~p & y).sum())
        r1 = a / n1 if n1 else np.nan
        r0 = b / n0 if n0 else np.nan
        result[farm] = dict(rows=len(group), proxy_count=n1, drop_count=int(y.sum()),
                            overlap=a, risk_proxy=r1, risk_nonproxy=r0,
                            risk_ratio=r1/r0 if r0 else np.nan)
    return result


def main():
    x = pd.read_csv(DATA / 'train_X.csv', usecols=COLS)
    y = pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    frame = x.merge(y, on='row_id', validate='one_to_one')
    frame['farm'] = frame.row_id.str[:3]
    frame = frame[frame.farm.isin(('F13', 'F47'))].copy()
    frame['day'] = frame.row_id.str[4:7].astype(int)
    frame['hour'] = frame.row_id.str[8:10].astype(int)
    frame.sort_values(['farm', 'day', 'hour'], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    assert len(frame) == 9600 and frame.groupby(['farm', 'day']).size().eq(24).all()
    for col in ['in_hum', 'in_temp', 'out_hum', 'act_fog', 'act_vent', 'sub_ec']:
        frame['d_' + col] = frame.groupby(['farm', 'day'])[col].diff()
        frame['prev_' + col] = frame.groupby(['farm', 'day'])[col].shift()
    frame = frame[frame.hour > 0].copy().reset_index(drop=True)
    assert len(frame) == 9200
    ok = (frame.act_fog.le(5) & frame.prev_act_fog.le(5) &
          frame.act_vent.le(5) & frame.prev_act_vent.le(5))
    p1 = (frame.d_in_hum.ge(5) & ok).fillna(False).to_numpy(bool)
    p2 = (p1 & frame.d_in_temp.abs().le(1).to_numpy(bool) &
          frame.d_out_hum.lt(5).to_numpy(bool))
    drop = frame.d_sub_ec.le(-.08).fillna(False).to_numpy(bool)
    proxies = {'P1': p1, 'P2': p2}
    observed = {k: summaries(frame, p, drop) for k, p in proxies.items()}
    frame['section'] = np.where(frame.day < 179, 0, 1)
    # Permute complete 23-hour day outcome curves within farm and section.
    day_ids = frame[['farm', 'section', 'day']].drop_duplicates().reset_index(drop=True)
    day_lookup = {tuple(v): i for i, v in enumerate(day_ids.itertuples(index=False, name=None))}
    matrix = np.zeros((len(day_ids), 23), bool)
    row_day = np.empty(len(frame), int)
    for i, row in enumerate(frame.itertuples()):
        d = day_lookup[(row.farm, row.section, row.day)]
        row_day[i] = d
        matrix[d, row.hour - 1] = drop[i]
    strata = [g.index.to_numpy() for _, g in day_ids.groupby(['farm', 'section'])]
    rng = np.random.default_rng(20260929)
    permuted_overlap = {k: np.empty(2000, int) for k in proxies}
    for b in range(2000):
        perm = np.arange(len(day_ids))
        for idx in strata:
            perm[idx] = rng.permutation(idx)
        shuffled = matrix[perm[row_day], frame.hour.to_numpy(int) - 1]
        for k, p in proxies.items():
            permuted_overlap[k][b] = int(np.sum(p & shuffled))
    inference = {}
    for k, p in proxies.items():
        overlap = observed[k]['ALL']['overlap']
        ratio = observed[k]['ALL']['risk_ratio']
        q = float((1 + np.sum(permuted_overlap[k] >= overlap)) / 2001)
        both = all(observed[k][f]['risk_ratio'] > 1.5 for f in ('F13','F47'))
        inference[k] = dict(permutation_p_ge=q,
                            null_overlap_q95=float(np.quantile(permuted_overlap[k], .95)),
                            passes_screen=bool(overlap >= 20 and ratio > 2 and both and q < .005))
    result = dict(observed=observed, inference=inference,
                  progress_to_model=any(z['passes_screen'] for z in inference.values()),
                  source='public train_X/train_y only', hidden_labels_read=False)
    out = ROOT / 'local/ec_irrigation_proxy' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
