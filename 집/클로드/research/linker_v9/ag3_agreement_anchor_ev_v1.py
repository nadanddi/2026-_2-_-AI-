# -*- coding: utf-8 -*-
"""AG3 (2026-10-12 집 클로드; 사용자: '애매한 날짜·동을 다른 모델이 확인해 개선').  사전 고정(이 파일 커밋 후 실행).
AG2(서술): v7 과 DF(날짜 먼저 + 0시 지문)가 같은 전날을 고르면 맞음 — v7 앵커 중 일치 97%(36) vs 불일치 54%(24), v7 비앵커(정답 기록 선택) 중 일치 100%(12) vs 7%(73).
규칙(조정할 수치 없음): '두 방법 일치' = v7 거울 CV 선택 prev == DF 선택(df1_picks_mirror.csv; DF 규칙은 DF1 그대로, margin·거리 조건 없음).
  FILT: v7 앵커(p ≥ .8, 정답, |dT_air| ≤ 1.5) 중 일치하는 것만 유지.
  ADD : v7 앵커 + (v7 비앵커이고 v7 선택이 정답 기록이며 일치, |dT_air| ≤ 1.5) 추가.
  BOTH: FILT + ADD.
점수: EV, 기준 = 13회차 구성(온도 v22, EC EA2) — 같은 공식·차단(DF1 과 동일), 전날이 그 EV 폴드 val/del 이면 버림.
  온도 각 안: v18 위에 앵커 집합 전체를 다시 얹음(q_h + BETA_h clip(A − .36 − q0, ±2)); EC: SG2 위에 EA2 공식.
채택(안마다·열마다, 본페로니 3안 → α .025/3 = .0083): 모든 칸(온도 4, EC 3) 전체 val RMSE 개선 AND 2차 val RMSE 개선(변화 없으면 통과)
  AND 기록 부트스트랩 P(악화) < .0083 (모든 칸) AND 상위 1기록 몫 ≤ .5 (칸 평균).
서술: 각 안의 앵커 수·진실 정확도; 평가 60 에 적용 시 빠지는/더해지는 기록.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
H = r"C:\work\farmai\집\클로드\research"
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv")).set_index("rec")
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
V["df_prev"] = P.prev.reindex(V.index); V["agree"] = V.prev == V.df_prev; V["lab"] = V.prev_is_labelled.astype(bool)
v7a = (V.p >= .8) & V.lab & (V.dT_air.abs() <= 1.5)
add = (~v7a) & V.lab & V.agree & (V.dT_air.abs() <= 1.5)
ARMS = {"V7": v7a, "FILT": v7a & V.agree, "ADD": v7a | add, "BOTH": (v7a & V.agree) | add}
for k, m in ARMS.items():
    S = V[m]; t = [TP.get(r, "") for r in S.index]; ok = [p == x for p, x in zip(S.prev, t) if x]
    print(f"{k:5s}: anchors {int(m.sum())}, truth-known acc {np.mean(ok):.3f} (n {len(ok)}), pass-2 {int((S.index.str[4:7].astype(int) >= 179).sum())}")
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
def temp_pred(c, mask):
    q18 = D[f"v18_a_{c}"].values; out = q18.copy()
    for rec, G in D.groupby("rec"):
        if rec not in V.index or not mask.get(rec, False): continue
        a = idx[V.at[rec, "prev"]]; k = int(G.fold.iloc[0])
        if V.at[rec, "prev"] in hid[k]: continue
        ix = G.index.values; q0 = q18[ix[G.hour.values == 0]][0]
        out[ix] = q18[ix] + BETA[G.hour.values] * np.clip(ST[a, 23] - .36 - q0, -2, 2)
    return out
res = []
for c in ["7A", "7B", "101A", "101B"]:
    base = D[f"v22_a_{c}"].values
    chk = temp_pred(c, ARMS["V7"].to_dict()); print(f"temp cell {c}: rebuilt V7 vs stored v22 max diff {np.abs(chk - base).max():.4f}")
    for k in ("FILT", "ADD", "BOTH"):
        res.append(dict(col="TEMP", arm=k, cell=c, **score(D, base, temp_pred(c, ARMS[k].to_dict()))))
for sd, G0 in R.groupby("seed"):
    G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values
    cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
    for k in ("FILT", "ADD", "BOTH"):
        new = G0.SG2.values.copy(); mask = ARMS[k].to_dict()
        for rec, G in G0.groupby("rec"):
            if rec not in V.index or not mask.get(rec, False) or V.at[rec, "prev"] in hid[int(G.fold.iloc[0])]: continue
            ix = G.index.values; new[ix] = np.clip(G0.SG2.values[ix] + (EC[idx[V.at[rec, "prev"]], 23] - cm[ix]), lo, hi)
        res.append(dict(col="EC", arm=k, cell=str(sd), **score(G0, base, new)))
S = pd.DataFrame(res); S["all_%"] = 100 * (S.all_new / S.all_base - 1); S["p2_%"] = 100 * (S.p2_new / S.p2_base - 1)
pd.set_option("display.width", 230); print(S.round(4).to_string(index=False))
for (col, k), g in S.groupby(["col", "arm"]):
    ok = (g.all_new < g.all_base).all() and (g.p2_new <= g.p2_base).all() and (g.P_worse < .025 / 3).all() and g.top_share.mean() <= .5
    print(f"{col} {k}: {'ADOPT' if ok else 'FAIL'}")
# eval preview
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
N4 = pd.read_csv(os.path.join(H, "na4_v1.csv")).set_index("rec")
CAL = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
def dfpick(b):
    vi = np.where(TRAIN)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    if len(ds) == 0: return None
    d = pd.Series(ds).mode().iloc[0]; cand = [a for a in vi if DAY[a] == d - 1 and not any(TRAIN[s] and DAY[s] == d for s in SUCC.get(a, []))]
    return rid(min(cand, key=lambda a: fpd(b, a))) if cand else None
print("\neval: v7 choice vs DF pick")
for r, c in E7.iterrows():
    b = idx[r]; dp = dfpick(b); a7 = c.p_h0 >= .8 and bool(c.prev_is_labelled) and abs(c.dT_air) <= 1.5
    ag = dp == c.prev_h0
    tag = ("KEEP" if ag else "DROP(FILT)") if a7 else ("ADD" if (ag and bool(c.prev_is_labelled) and abs(c.dT_air) <= 1.5) else "")
    if a7 or tag: print(f"   {r} v7 {c.prev_h0} p {c.p_h0:.2f} | DF {dp} | {tag}")
