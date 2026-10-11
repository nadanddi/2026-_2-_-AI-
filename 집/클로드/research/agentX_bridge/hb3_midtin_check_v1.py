# -*- coding: utf-8 -*-
"""hb3: where does the mid_Tin signal (temp, k>=2) come from? correlations by pass and k,
with/without gap; partial r after gap; per-farm."""
import os, sys
sys.path.insert(0, r'C:\work\farmai\집\클로드\research')
import env
import numpy as np, pandas as pd
OUT = r'C:\work\farmai\집\클로드\research\agentX_bridge'
for target in ['temp', 'ec']:
    P = pd.read_csv(os.path.join(OUT, f'hb1_pairs_{target}.csv'))
    P['farm'] = np.where(P.t < 230, 'F13?', 'F47?')
    print('=====', target)
    for k in range(2, 6):
        for ps in [0, 1]:
            s = P[(P.k == k) & (P.pass2 == ps)]
            out = []
            for y in ['r0', 'rm']:
                c = np.corrcoef(s.mid_Tin, s[y])[0, 1]
                X = np.c_[np.ones(len(s)), s.gap]; e = s[y] - X @ np.linalg.lstsq(X, s[y], rcond=None)[0]
                pc = np.corrcoef(s.mid_Tin, e)[0, 1]
                out.append(f'{y}: r {c:+.2f} partial|gap {pc:+.2f}')
            print(f'  k={k} pass{ps+1} n={len(s)}  ' + '  '.join(out) + f'  corr(mid_Tin,gap) {np.corrcoef(s.mid_Tin, s.gap)[0,1]:+.2f}')
