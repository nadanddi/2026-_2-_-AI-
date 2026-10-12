# -*- coding: utf-8 -*-
"""AG4 (2026-10-12 집 클로드; 사용자: '날짜·동만 잇는 좋은 모델 몇 개로 소프트/하드 보팅').  사전 고정(이 파일 커밋 후 실행).
모델 4개(LM1, 거울 CV 0시 판단): A v7 / B DF(날짜 먼저 + 0시 지문, p=1) / C 자정연속 최근접 / D LightGBM 쌍 분류.
표: 모델마다 1순위 전날과 확률 p.  DF 가 판단 못 한 기록은 기권.
  HV3 : 4표 중 3표 이상이 같은 '정답 기록' 전날을 고르면 그 전날로 앵커(v7 앵커를 대체).
  ADD3: v7 앵커 ∪ HV3 (겹치면 v7 그대로).
  SV  : 후보마다 점수 = Σ_모델 p(그 모델이 그 후보를 골랐을 때) / 4, 최고 후보가 정답 기록이고 점수 ≥ .70 이면 앵커(v7 대체).
  공통: |dT_air| ≤ 1.5 차단 유지(같은 23→0시 실내 차 = IN_[b,0,0] − IN_[a,23,0]).
점수·판정: AG3 와 같음(EV, 기준 13회차 v22/EA2, 같은 공식, 전날이 EV 폴드 val/del 이면 버림); 3안 본페로니 α .025/3.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
H = r"C:\work\farmai\집\클로드\research"
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
V["lab"] = V.prev_is_labelled.astype(bool)
LM = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "lm1_choices.csv"))
idx0 = {rid(i): i for i in range(N)}
dT = lambda r, p: float(IN_[idx0[r], 0, 0] - IN_[idx0[p], 23, 0])
v7a = (V.p >= .8) & V.lab & (V.dT_air.abs() <= 1.5)
hv, sv = {}, {}
for r, G in LM.groupby("rec"):
    cnt = G.prev.value_counts(); top = cnt.index[0]
    if cnt.iloc[0] >= 3 and TRAIN[idx0[top]] and abs(dT(r, top)) <= 1.5: hv[r] = top
    s = G.groupby("prev").p.sum() / 4; t2 = s.idxmax()
    if s.max() >= .70 and TRAIN[idx0[t2]] and abs(dT(r, t2)) <= 1.5: sv[r] = t2
V = V.reindex(sorted(set(V.index) | set(hv) | set(sv)))
def arm(prevmap, keep_v7):
    out = {}
    if keep_v7: out.update({r: V.at[r, "prev"] for r in V.index[v7a.reindex(V.index).fillna(False)]})
    for r, p in prevmap.items(): out.setdefault(r, p)
    return out
ARMSP = {"V7": arm({}, True), "HV3": arm(hv, False), "ADD3": arm(hv, True), "SV": arm(sv, False)}
ARMS = {k: pd.Series({r: True for r in d}) for k, d in ARMSP.items()}
PREV = lambda k, r: ARMSP[k][r]
for k, d in ARMSP.items():
    ok = [p == TP.get(r, "") for r, p in d.items() if TP.get(r, "")]
    print(f"{k:5s}: anchors {len(d)}, truth-known acc {np.mean(ok):.3f} (n {len(ok)}), pass-2 {sum(int(r[4:7]) >= 179 for r in d)}")
idx = {rid(i): i for i in range(N)}
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy(); lo, hi = R.y.min(), R.y.max()
rng = np.random.default_rng(20261014); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def score(G, base, new):
    y = G.y.values; recs = G.rec.values; p2 = G.day.values >= 179
    db = pd.Series((base - y) ** 2).groupby(recs).sum(); dn = pd.Series((new - y) ** 2).groupby(recs).sum(); dl = (dn - db).values
    pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
    return dict(all_base=rm(base - y), all_new=rm(new - y), p2_base=rm((base - y)[p2]), p2_new=rm((new - y)[p2]), P_worse=pw,
                top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan, changed=int((np.abs(dl) > 1e-12).sum()))
def temp_pred(c, mask, k_arm="V7"):
    q18 = D[f"v18_a_{c}"].values; out = q18.copy()
    for rec, G in D.groupby("rec"):
        if not mask.get(rec, False): continue
        pv = PREV(k_arm, rec); a = idx[pv]; k = int(G.fold.iloc[0])
        if pv in hid[k]: continue
        ix = G.index.values; q0 = q18[ix[G.hour.values == 0]][0]
        out[ix] = q18[ix] + BETA[G.hour.values] * np.clip(ST[a, 23] - .36 - q0, -2, 2)
    return out
res = []
for c in ["7A", "7B", "101A", "101B"]:
    base = D[f"v22_a_{c}"].values
    chk = temp_pred(c, ARMS["V7"].to_dict()); print(f"temp cell {c}: rebuilt V7 vs stored v22 max diff {np.abs(chk - base).max():.4f}")
    for k in ("HV3", "ADD3", "SV"):
        res.append(dict(col="TEMP", arm=k, cell=c, **score(D, base, temp_pred(c, ARMS[k].to_dict(), k))))
for sd, G0 in R.groupby("seed"):
    G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values
    cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
    for k in ("HV3", "ADD3", "SV"):
        new = G0.SG2.values.copy(); mask = ARMS[k].to_dict()
        for rec, G in G0.groupby("rec"):
            if not mask.get(rec, False) or PREV(k, rec) in hid[int(G.fold.iloc[0])]: continue
            ix = G.index.values; new[ix] = np.clip(G0.SG2.values[ix] + (EC[idx[PREV(k, rec)], 23] - cm[ix]), lo, hi)
        res.append(dict(col="EC", arm=k, cell=str(sd), **score(G0, base, new)))
S = pd.DataFrame(res); S["all_%"] = 100 * (S.all_new / S.all_base - 1); S["p2_%"] = 100 * (S.p2_new / S.p2_base - 1)
pd.set_option("display.width", 230); print(S.round(4).to_string(index=False))
for (col, k), g in S.groupby(["col", "arm"]):
    ok = (g.all_new < g.all_base).all() and (g.p2_new <= g.p2_base).all() and (g.P_worse < .025 / 3).all() and g.top_share.mean() <= .5
    print(f"{col} {k}: {'ADOPT' if ok else 'FAIL'}")
