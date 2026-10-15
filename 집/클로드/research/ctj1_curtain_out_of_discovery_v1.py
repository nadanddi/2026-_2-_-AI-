# -*- coding: utf-8 -*-
"""CTJ1 (2026-10-15 집 클로드; CTX1 분해 후속).  사전 고정(커밋 후 실행).  EC.  학습 없음(CT2 저장 행).
CTX1: 커튼 템플릿 특징(THS5) 이득은 템플릿 일치 ≥ .8 기록에 몰림, 발견에 쓴 2차 4일(F13_214·217, F47_229·231)이 대부분이나 빼도 세 검증기 모두 음(−2.6/−3.3/−1.0).
판정(발견 밖 근거, 단일 안 α .025): 발견 4일을 뺀 2차 행에서 pre(SG2 전) THS5 vs REF — 3검증기 x 3시드 9칸 모두 dSSE < 0
  & P2LOO 블록(농장 x 5일) P(악화) < .025 & 손해 기록-시드(P2LOO·EL1) 중 상위 1기록 빼도 9칸 유지.
서술: 고EC/보통, 템플릿 대역.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ctj1_curtain_out_of_discovery_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
CK = os.path.join(env.LOCAL, "ct2_ckpt"); SEEDS = (8383, 1919, 7171); DISC = {"F13_214", "F13_217", "F47_229", "F47_231"}
D = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True); D["rec"] = D.row_id.str[:7]
D = D[(D.day >= 179) & ~D.rec.isin(DISC)]
cells, blocks = [], None
for v in ("P2LOO", "EL1", "DIAG10"):
    g = D[D.validator == v]
    for s in SEEDS:
        dd = (g[f"pre_THS5_{s}"] - g.sub_ec) ** 2 - (g[f"pre_REF_{s}"] - g.sub_ec) ** 2
        rec = dd.groupby(g.rec).sum(); top = rec.idxmin()
        rel = 100 * (np.sqrt(((g[f"pre_THS5_{s}"] - g.sub_ec) ** 2).mean()) / np.sqrt(((g[f"pre_REF_{s}"] - g.sub_ec) ** 2).mean()) - 1)
        cells.append(dict(val=v, seed=s, dSSE=float(dd.sum()), rel=rel, dSSE_wo_top=float(dd.sum() - rec[top]), top=top))
    if v == "P2LOO":
        dm = np.mean([(g[f"pre_THS5_{s}"] - g.sub_ec) ** 2 - (g[f"pre_REF_{s}"] - g.sub_ec) ** 2 for s in SEEDS], axis=0)
        blocks = pd.Series(dm, index=g.index).groupby((g.farm + "_" + (g.day // 5).astype(str)).values).mean().values
C = pd.DataFrame(cells); print(C.round(3).to_string(index=False))
P = float((blocks[np.random.default_rng(20270127).integers(0, len(blocks), (20000, len(blocks)))].mean(1) >= 0).mean())
ok = bool((C.dSSE < 0).all()) and P < .025 and bool((C.dSSE_wo_top < 0).all())
print(f"CTJ1: 9 cells better {bool((C.dSSE < 0).all())} | P2LOO P {P:.4f} | without top record still 9/9 {bool((C.dSSE_wo_top < 0).all())} -> {'PASS' if ok else 'FAIL'}")
