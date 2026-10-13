# -*- coding: utf-8 -*-
"""TL1-T3 (2026-10-14 집 클로드; 비평 T3 '평가 노출 지도').  서술(판정 아님, 정답 없음).  EC.
비평(6.870): TL1 이득은 과대예측 보통 3기록에서만 → 평가에서 몇 기록이 얼마나 움직이는지가 TL1 을 더 할지 정함.
REF = 전체 학습 자료(ct2.test_shift 와 같은 학습 행), NEAR = 2차 날 + 학습 달력(DMT3e)으로 어떤 학습 2차 날과 같은 구간·날짜 차 <= 2 인 1차 날(평가 날짜 안 씀; TL1 과 같은 정의).
pre = clip(shrink(.8 R3 + .2 PFN)) 시드 2121·3434·5656 평균, PFN 문맥 13~16.  SG2 는 적용 안 함(한계: 차이만 봄).
관문 = REF pre 0..h 누적평균 >= 1.0 → REF, 아니면 NEAR.  출력: 평가 기록별 하루 평균 Δ(TL − REF), '내리기만' Δ(min), |Δ| > .1 인 기록 수.
결정 규칙(미리 정함): '내리기만' 하루 평균 Δ <= −.10 인 평가 기록이 3개 미만이면 TL1 계열(T2) 중단.
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "1"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
wt0, dc4, p3, core, dp1, ct1 = ct2.wt0, ct2.dc4, ct2.p3, ct2.core, ct2.dp1, ct2.ct1
SEEDS = (2121, 3434, 5656); CTX = (13, 14, 15, 16); W = (.6, .3, .1)
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s_, d_) for r, s_, d_ in zip(CAL.rec, CAL.seg, CAL.day)}
key = lambda f, d: f"{f}_{int(d):03d}"
raw, full, lab, lock, fds = ct2.prep()
tX, ty, sX = __import__("common").load_raw()
st = sX[sX.row_id.str[:3].isin(["F13", "F47"])][["row_id"] + list(p3.RAW14)].reset_index(drop=True)
fullT = core.identify(st)
T = core.features(st).merge(fullT[["row_id"] + list(p3.RAW14[:4])], on="row_id", validate="one_to_one")
T = T.join(pd.concat([dp1.day_feats(g) for _, g in T.groupby(["farm", "day"])]))
T = T.merge(ct1.thermal_features(fullT), on="row_id", how="left", validate="one_to_one")
wv = dc4.weather_vectors(pd.concat([full, fullT], ignore_index=True))
tr = lab.copy(); tdays = tr[["farm", "day"]].drop_duplicates(); vdays = T[["farm", "day"]].drop_duplicates().reset_index(drop=True)
season, vq = dc4.season_index(tdays, vdays, wv)
tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
T = T.merge(vdays, on=["farm", "day"], how="left").sort_values(["farm", "day", "hour"]).reset_index(drop=True)
labset = set(tr.groupby(["farm", "day"]).size().index)
p2dates = [DAY[key(f, d)] for f, d in labset if d >= 179 and key(f, d) in DAY]
near1 = {k for k, (s_, d_) in DAY.items() if int(k[4:]) < 179 and any(s_ == s2 and abs(d_ - d2) <= 2 for s2, d2 in p2dates)}
nm = np.array([(d >= 179) or (key(f, d) in near1) for f, d in zip(tr.farm, tr.day)]); trN = tr[nm].copy()
print(f"train rows {len(tr)} days {len(labset)} | NEAR days {trN[['farm', 'day']].drop_duplicates().shape[0]} | test rows {len(T)}", flush=True)
cols, PF = ct2.feature_sets(); FS, BS = cols["REF"]; lo, hi = tr.sub_ec.min(), tr.sub_ec.max(); Xq = T[PF].to_numpy(np.float32)
pre = {}
for lay, D in (("REF", tr), ("NEAR", trN)):
    Xt, yt = D[PF].to_numpy(np.float32), D.sub_ec.to_numpy(float)
    pf = np.mean([ct2.pfn(Xt[np.random.default_rng(c).choice(len(D), min(2000, len(D)), replace=False)], yt[np.random.default_rng(c).choice(len(D), min(2000, len(D)), replace=False)], Xq, c) for c in CTX], axis=0)
    ps = []
    for s in SEEDS:
        e, l, m = wt0.members(D, T, s, FS, BS); ps.append(np.clip(ct2.shrink(.8 * (W[0] * e + W[1] * l + W[2] * m) + .2 * pf, T), lo, hi))
    pre[lay] = np.mean(ps, 0); print(lay, "done", flush=True)
T["ref"], T["near"] = pre["REF"], pre["NEAR"]
T["cm"] = T.groupby(["farm", "day"]).ref.transform(lambda x: x.expanding().mean())
T["tl"] = np.where(T.cm >= 1.0, T.ref, T.near); T["tl_low"] = np.where(T.cm >= 1.0, T.ref, np.minimum(T.ref, T.near))
T["rec"] = [key(f, d) for f, d in zip(T.farm, T.day)]
S = T.groupby("rec").agg(ref=("ref", "mean"), near=("near", "mean"), d_tl=("tl", "mean"), d_low=("tl_low", "mean"), routed_ref=("cm", lambda x: float((x >= 1).mean())))
S["d_tl"] -= S.ref; S["d_low"] -= S.ref
pd.set_option("display.width", 200); print(S.round(3).sort_values("d_low").to_string())
n_big = int((S.d_low <= -.10).sum())
print(f"\nrecords with lower-only day-mean shift <= -.10: {n_big} | |TL shift| > .10: {int((S.d_tl.abs() > .10).sum())} | mean TL shift {S.d_tl.mean():+.3f} | rows moved RMS (TL) {np.sqrt(np.mean((T.tl - T.ref) ** 2)):.3f}")
print("T3 decision:", "continue to T2" if n_big >= 3 else "stop TL1 line (fewer than 3 records move by >= .10)")
S.to_csv(os.path.join(HERE, "tl1t3_eval_exposure_v1.csv"))
