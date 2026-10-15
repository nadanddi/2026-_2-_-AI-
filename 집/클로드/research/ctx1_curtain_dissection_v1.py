# -*- coding: utf-8 -*-
"""CTX1 (2026-10-15 집 클로드; 사용자 '뜯어봐서 살릴 부분만' — CT1/CT2 커튼 일정 특징).  서술 분해(판정 없음), 실행 전 커밋.  EC.
CT2(6.464) 저장 행 local/ct2_ckpt: pre(SG2 전 = 27번 EC 의 pre 단계와 같은 위치) REF vs THS5, 3시드, DIAG10·EL1·P2LOO.
CT1 문서: 템플릿(0~8시 보온커튼 ≤ .1, 9시·16시 50~95, 10~15시 ≥ 99.9, 17~23시 ≤ .1)은 고EC 날 탐색(EH3/EH4)에서 찾음(사후) — 발견에 쓴 2차 패턴 날 F13_214·217, F47_229·231.
측정: (1) 기록별 pre THS5 − REF dSSE(시드 평균), 상위 이득·손해 기록과 그 날의 템플릿 일치율(하루 전체, 서술용)·고EC 여부.
      (2) 템플릿 일치율 대역별(≥ .8 / .5~.8 / < .5) dSSE 합 — 이득이 패턴 날에 있는가, 발견 날 4개를 빼면 남는가.
      (3) 평가 60기록(test_X 입력만)의 템플릿 일치율 분포 — 평가에서 이 특징이 작동할 기록 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ctx1_curtain_dissection_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CK = os.path.join(env.LOCAL, "ct2_ckpt")
SEEDS = (8383, 1919, 7171); DISC = {"F13_214", "F13_217", "F47_229", "F47_231"}
def template_share(X):
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int); th = X.act_thermal.values; h = X.h.values
    ok = np.where(h <= 8, th <= .1, np.where((h == 9) | (h == 16), (th >= 50) & (th <= 95), np.where((h >= 10) & (h <= 15), th >= 99.9, th <= .1)))
    X["ok"] = np.where(np.isnan(th), np.nan, ok.astype(float)); return X.groupby("rec").ok.mean()
TR = template_share(pd.read_csv(os.path.join(env.DATA, "train_X.csv"))); TE = template_share(pd.read_csv(os.path.join(env.DATA, "test_X.csv")))
D = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True); D["rec"] = D.row_id.str[:7]
dd = np.mean([(D[f"pre_THS5_{s}"] - D.sub_ec) ** 2 - (D[f"pre_REF_{s}"] - D.sub_ec) ** 2 for s in SEEDS], axis=0); D["dd"] = dd
D["dm"] = D.groupby(["validator", "rec"]).sub_ec.transform("mean"); D["p2"] = D.day >= 179
pd.set_option("display.width", 200)
for v in ("P2LOO", "EL1", "DIAG10"):
    g = D[(D.validator == v) & D.p2].groupby("rec").agg(dd=("dd", "sum"), dm=("dm", "first")); g["tmpl"] = g.index.map(TR); g["disc"] = g.index.isin(DISC)
    band = pd.cut(g.tmpl, [-.01, .5, .8, 1.01], labels=["<.5", ".5-.8", ">=.8"])
    print(f"\n== {v} pass-2 records {len(g)}: total dSSE {g.dd.sum():+.3f} (negative = THS5 better) | without discovery days {g[~g.disc].dd.sum():+.3f}")
    print("  by template band:", g.groupby(band, observed=False).dd.agg(["size", "sum"]).round(3).to_dict("index"))
    print("  by EC level (day mean >= 1):", g.groupby(g.dm >= 1).dd.sum().round(3).to_dict())
    print("  top gains:", [(r, round(x, 3), round(t, 2), round(m, 2)) for r, x, t, m in g.nsmallest(6, "dd")[["dd", "tmpl", "dm"]].itertuples()])
    print("  top losses:", [(r, round(x, 3), round(t, 2), round(m, 2)) for r, x, t, m in g.nlargest(4, "dd")[["dd", "tmpl", "dm"]].itertuples()])
print("\n(3) evaluation records template share: >=.8", int((TE >= .8).sum()), "| .5-.8", int(((TE >= .5) & (TE < .8)).sum()), "| <.5", int((TE < .5).sum()), "| of", len(TE))
print("    training records template share >=.8:", int((TR >= .8).sum()), "of", len(TR), "| pass-2 training >=.8:", sorted(TR[(TR >= .8) & (TR.index.str[4:].astype(int) >= 179)].index))
print("    evaluation records >=.5:", sorted(TE[TE >= .5].round(2).items()))
