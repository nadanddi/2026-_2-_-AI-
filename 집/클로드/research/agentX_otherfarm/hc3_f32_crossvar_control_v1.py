# -*- coding: utf-8 -*-
"""양성 대조 2: out_rad(F32) 대 다른 온실 in_rad - 변수 종류가 달라도 같은 날짜·지역이면 잡히는지 (학습 기록만).
또 F13·F47 기준을 1차 구간(day<179) 기록 일차 순서로만 만든 판(달력 재구성 오류 영향 확인)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import env
import numpy as np, pandas as pd
X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
X['farm'] = X.row_id.str[:3]; X['day'] = X.row_id.str[4:7].astype(int); X['hour'] = X.row_id.str[8:10].astype(int)
def detr(s, w=15):
    s = s.sort_index(); full = s.reindex(range(s.index.min(), s.index.max() + 1))
    return (full - full.rolling(w, center=True, min_periods=5).median()).reindex(s.index)
def scan(a, b, lags, nmin):
    out = []
    for L in lags:
        bb = b.reindex(a.index + L).values; aa = a.values; ok = np.isfinite(aa) & np.isfinite(bb)
        if ok.sum() >= nmin: out.append((L, ok.sum(), np.corrcoef(aa[ok], bb[ok])[0, 1]))
    return pd.DataFrame(out, columns=['lag', 'n', 'r'])
ser = {}
for f, D in X.groupby('farm'):
    v = 'in_rad' if D.in_rad.notna().mean() > .5 else 'out_rad'
    ser[f] = detr(D.pivot_table(index='day', columns='hour', values=v, aggfunc='mean').sum(axis=1, min_count=20))
for refname in ['F32', 'F13', 'F47']:
    ref = ser[refname]
    if refname in ('F13', 'F47'):
        ref = ref[ref.index < 179]  # 1차 구간: 기록 순서 = 달력 순서(날짜당 1~2기록)
    rows = []
    for f in sorted(ser):
        if f in ('F13', 'F47', 'F32'): continue
        R = scan(ref, ser[f], range(-400, 401), 60)
        j = R.r.idxmax(); rows.append((f, int(R.lag[j]), int(R.n[j]), R.r[j], R[(R.lag - R.lag[j]).abs() > 3].r.max()))
    S = pd.DataFrame(rows, columns=['farm', 'lag', 'n', 'r_best', 'r_2nd']).sort_values('r_best', ascending=False)
    print(f'\n기준 {refname} ({"out_rad" if refname!="F32" else "out_rad"}) 상위 8:'); print(S.head(8).round(3).to_string(index=False))
    print('r_best 중앙', S.r_best.median().round(3), '| r_best>=0.7 온실 수', (S.r_best >= .7).sum())
