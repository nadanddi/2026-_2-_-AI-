# -*- coding: utf-8 -*-
"""EVHP (2026-10-12 집 클로드; 사용자: "HP2+메타 일치 앵커로 검증에서 채점").  사전 고정(이 파일 커밋 후 실행).
앵커 집합 = local/linker_v10/hp3_anchor_set.csv (HP2 혼합 규칙 선택 == META2 선택 AND 메타 확률 ≥ .5, |dT_air| ≤ 1.5 는 HP3 에서 이미 적용).  나머지(공식·기준·판정)는 아래 EV10 설명과 같음.
v7 앵커(13회차 구성) 대신 짝찾기 v10 거울 CV 선택(cv_all400_choice_v10m)으로 앵커를 붙여 EV 채점.  앵커 규칙·공식은 13회차 그대로:
  앵커 = p ≥ .8 AND 전날 정답 기록 AND |dT_air| ≤ 1.5, 전날이 그 EV 폴드 val/del 이면 버림.
  온도: q18_h + BETA_h clip(A_st − .36 − q0, ±2) (BETA = ta8_beta, 13회차 상수 그대로 — 재측정은 따로).  EC: SG2_h + (A_ec − SG2 0..h 누적평균), 학습 범위로 자름.
기준 = 저장된 13회차 구성(온도 v22 4칸, EC EA2 3시드).  열마다 한 안(α .025):
  모든 칸 전체 val RMSE 개선 AND 2차 val RMSE 개선(또는 같음) AND 기록 단위 부트스트랩(4000) P(악화) < .025 AND 이득 상위 1기록 몫 ≤ .5 (칸 평균).
서술: v7·v10 앵커 집합 차이(EV 에 실제 적용된 기록), 1·2차별 변화.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
H = r"C:\work\farmai\집\클로드\research"
idx = {rid(i): i for i in range(N)}
def anchors(path):
    V = pd.read_csv(path, encoding="utf-8-sig")
    V = V[(V.p >= .8) & V.prev_is_labelled.astype(bool) & (V.dT_air.abs() <= 1.5)]
    return dict(zip(V.rec, V.prev))
A7 = anchors(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")); _H = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "hp3_anchor_set.csv")); A10 = dict(zip(_H.rec, _H.pick))
print(f"mirror anchors v7 {len(A7)} | v10 {len(A10)} | same prev {sum(1 for r in A7 if A10.get(r) == A7[r])}")
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy(); lo, hi = R.y.min(), R.y.max()
rng = np.random.default_rng(20261018); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def score(G, base, new):
    y = G.y.values; recs = G.rec.values; p2 = G.day.values >= 179
    db = pd.Series((base - y) ** 2).groupby(recs).sum(); dn = pd.Series((new - y) ** 2).groupby(recs).sum(); dl = (dn - db).values
    pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
    return dict(all_base=rm(base - y), all_new=rm(new - y), p2_base=rm((base - y)[p2]), p2_new=rm((new - y)[p2]), P_worse=pw,
                top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan, changed=int((np.abs(dl) > 1e-12).sum()))
def temp_pred(c, A):
    q18 = D[f"v18_a_{c}"].values; out = q18.copy(); used = set()
    for rec, G in D.groupby("rec"):
        if rec not in A or A[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; q0 = q18[ix[G.hour.values == 0]][0]
        out[ix] = q18[ix] + BETA[G.hour.values] * np.clip(ST[idx[A[rec]], 23] - .36 - q0, -2, 2); used.add(rec)
    return out, used
res = []; used7 = used10 = set()
for c in ["7A", "7B", "101A", "101B"]:
    base = D[f"v22_a_{c}"].values
    chk, used7 = temp_pred(c, A7); print(f"temp {c}: rebuilt v7 vs stored v22 max diff {np.abs(chk - base).max():.4f}")
    new, used10 = temp_pred(c, A10); res.append(dict(col="TEMP", cell=c, **score(D, base, new)))
print(f"EV-applied anchors (temp): v7 {len(used7)} | v10 {len(used10)} | only v7 {len(used7 - used10)} | only v10 {len(used10 - used7)}")
for sd, G0 in R.groupby("seed"):
    G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values
    cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; new = G0.SG2.values.copy()
    for rec, G in G0.groupby("rec"):
        if rec not in A10 or A10[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; new[ix] = np.clip(G0.SG2.values[ix] + (EC[idx[A10[rec]], 23] - cm[ix]), lo, hi)
    res.append(dict(col="EC", cell=str(sd), **score(G0, base, new)))
S = pd.DataFrame(res); S["all_%"] = 100 * (S.all_new / S.all_base - 1); S["p2_%"] = 100 * (S.p2_new / S.p2_base - 1)
pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
for col, g in S.groupby("col"):
    ok = (g.all_new < g.all_base).all() and (g.p2_new <= g.p2_base).all() and (g.P_worse < .025).all() and g.top_share.mean() <= .5
    print(f"{col}: {'ADOPT' if ok else 'FAIL'}")
