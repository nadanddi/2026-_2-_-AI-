# -*- coding: utf-8 -*-
"""AF2 (2026-10-12 집 클로드; AF1 후속).  사전 고정 — 실행 전 커밋.
규칙: 앵커 기록 중 운영 지문 거리 fp0(대상 0시 vs 전날 같은 시각 비교, linker v6 fp_dist) 가 문턱 t 보다 크면 앵커를 떼고 기본 예측 사용.
문턱 선택: EV 앵커 54 에서 t ∈ {.3, .4, .5, .6, .7, .8, .9, 1.0, 무한} 중 '앵커 순 ΔSSE 합'(필터 후) 최소 → 고른 t 를 DIAG10 앵커 34 에 적용해 판정.
판정(DIAG10): 필터판이 필터 없는 앵커판보다 3시드 모두 좋음(앵커 기록 RMSE) AND 기록 부트스트랩 P(worse) < .05(필터로 바뀌는 기록 수가 적어 .05 사용 — 사전 명시)
  AND 필터판이 기본(sg) 보다도 3시드 모두 좋음.  EV(선택에 씀)는 참고로만.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
D = pd.read_csv(os.path.join(here, "af1_v1.csv"))
TS = [.3, .4, .5, .6, .7, .8, .9, 1.0, np.inf]
E = D[D.world == "EV"]; tot = {t: float(E.dsse[E.fp0 <= t].sum()) for t in TS}
print("EV: net anchor dSSE after filter by threshold:", {str(t): round(v, 2) for t, v in tot.items()})
tbest = min(TS, key=lambda t: tot[t]); print("chosen threshold:", tbest)
# DIAG10 judgement per seed
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
Dd = D[D.world == "DIAG10"].set_index("rec"); keep = Dd.index[Dd.fp0 <= tbest]; drop = Dd.index[Dd.fp0 > tbest]
print("DIAG10 anchors:", len(Dd), "| dropped by filter:", list(drop))
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261249)
res = []; per = []
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); cm = Q.cumsum(1) / np.arange(1, 25)
    A_all, A_f = Q.copy(), Q.copy()
    for r in Dd.index:
        v = Q.loc[r].values + (EC23[L.at[r, "prev"]] - cm.loc[r].values)
        A_all.loc[r] = v
        if r in keep: A_f.loc[r] = v
    recs = list(Dd.index); Y = YD.loc[recs]
    res.append(dict(seed=sd, base=rm((Q.loc[recs] - Y).values), anchor_all=rm((A_all.loc[recs] - Y).values), anchor_filtered=rm((A_f.loc[recs] - Y).values)))
    per.append(((A_f.loc[recs] - Y) ** 2).sum(1) - ((A_all.loc[recs] - Y) ** 2).sum(1))
R = pd.DataFrame(res); print(R.round(4).to_string(index=False))
pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(pr), (20000, len(pr))); pw = float((pr.values[ii].sum(1) >= 0).mean())
ok = (R.anchor_filtered < R.anchor_all).all() and pw < .05 and (R.anchor_filtered < R.base).all()
print(f"filtered vs all-anchor: per-record dSSE sum {pr.sum():+.3f}, P(worse) {pw:.4f} -> VERDICT AF2: {'PASS' if ok else 'FAIL'}")
print("dropped records' own dSSE (filtered - all):", pr[pr != 0].round(3).to_dict())
