# -*- coding: utf-8 -*-
"""VL2: single-record dominance in d=1 next cells (cert & number+calendar). diagnostic only."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
H = r"C:\work\farmai\집\클로드\research\hx8\harness\out_v1"
rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
D = pd.read_csv(H + r"\hx4_numcal_Sq_v1.csv"); G = D[(D.side == 'next') & (D.d == 1)]
X = G[G.rec != 'F47_132']; print('numcal next d1 w/o F47_132 n', len(X), 'q', round(rm(X.q - X.y), 3), 'E1', round(rm(X.E1 - X.y), 3), 'E4h', round(rm(X.E4h - X.y), 3))
C = pd.read_csv(H + r"\hx3_cases_cert_Sq_v1.csv")
for s, est in (('next', 'E4p'), ('prev', 'E1')):
    G = C[(C.side == s) & (C.d == 1)]
    G = G.assign(ds=(G[est] - G.y) ** 2 - (G.q - G.y) ** 2).sort_values('ds')
    print(s, est); print(G[['rec', 'comp', 'y', 'q', 'anc', est, 'ds']].round(3).head(5).to_string(index=False))
    X = G[G.rec != 'F47_132']; print(f'  cert {s} d1 w/o F47_132 n {len(X)} q {rm(X.q - X.y):.3f} {est} {rm(X[est] - X.y):.3f} better {(abs(X[est] - X.y) < abs(X.q - X.y)).mean():.2f}')
    # cells by F47 only (eval F47_227 is F47)
    X = G[G.rec.str.startswith('F47')]; print(f'  F47 only n {len(X)} comps {X.comp.nunique()} q {rm(X.q - X.y):.3f} {est} {rm(X[est] - X.y):.3f}')
# two-sided gap<=4 (claimed E4 -66%) concentration
G = C[(C.side == 'two') & (C.g <= 4)]
G = G.assign(ds=(G.E4 - G.y) ** 2 - (G.q - G.y) ** 2)
byr = G.groupby('rec').ds.sum().sort_values(); print('two gap<=4 E4 dSSE by rec top3', byr.head(3).round(3).to_dict(), 'total', round(byr.sum(), 3), 'nrec', len(byr))
