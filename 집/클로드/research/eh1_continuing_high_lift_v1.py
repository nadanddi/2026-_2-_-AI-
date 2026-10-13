# -*- coding: utf-8 -*-
"""EH1 (2026-10-14 집 클로드; 사용자: '진행해봐').  사전 고정(커밋 후 실행).  EC.  과거 정보만.  2차 기록만(EC 2차 전용 절차).
EH0(6.863): '같은 동 앞 기록(2일 이내)이 정답 고EC & 운영 지문 거리 <= 고EC 유지 쌍 최대' 인 학습 기록 19/19 진짜 고EC, 기본(OOF) 잔차 +.22.
규칙(시각 h, 그 배치에서 보이는 정보만):
  대상 날짜 = 0시 외기 4칸이 같은 보이는 정답 기록들의 학습 달력(DMT3e) 최빈 (d, s);
  후보 = 보이는 정답 기록 중 (d−1 또는 d−2, 같은 구간) 이고 하루 평균 EC >= 1.0(고EC);  각 후보의 0..h 운영 지문 거리(구동기·CO2 8열, 4 로 자름);
  발동 = 가장 가까운 후보가 확신(후보 1개거나 2등/1등 >= 1.5) AND 거리 <= KEEP_h (학습 cert 고EC 유지 쌍의 0..h 거리 최댓값 — 학습 정답으로만);
  발동 시각에 P_h = sg_h + .22 (EH0 값; 학습 쌍 전체에서 잰 값이라 검증 기록 일부 포함 — 숫자 1개, 누수 작음, 낙관 쪽).
기준 = 앵커 없는 기본 예측(rv1 CUR sg / EV SG2) — 지금 제출 EC(12회차 + B5)에 EA2 가 없음.
판정(EC 분야 배정 규칙, 링커 의존, k = 1): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 AND (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록 없음);
  (2) EV 3시드 모두 전체 행 dSSE <= 0;  (3) DIAG10 손해 감시.  서술: 발동 기록·실제 고EC 비율, 평가 적용 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261211); lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {int(b): int(a) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def dist(a, b, h): return float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
keep = [(a, b) for b, a in CERT.items() if np.isfinite(DM[a]) and np.isfinite(DM[b]) and DM[a] >= 1.0 and DM[b] >= 1.0]
KEEP = np.array([max(dist(a, b, h) for a, b in keep) for h in range(24)]); LIFT = .22
print(f"keep-high cert pairs {len(keep)} | KEEP_h at 0/9/23h: {KEEP[0]:.3f} {KEEP[9]:.3f} {KEEP[23]:.3f}", flush=True)
def tdate(b, vis):
    vi = np.where(vis)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
def fires(b, vis):
    t = tdate(b, vis)
    if t is None: return [False] * 24, None
    cand = [a for a in np.where(vis)[0] if a != b and SEGT[a] == t[1] and DAYT[a] in (t[0] - 1, t[0] - 2) and DM[a] >= 1.0]
    out, pick = [], None
    for h in range(24):
        if not cand: out.append(False); continue
        d = sorted((dist(a, b, h), a) for a in cand); ok = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= KEEP[h]
        out.append(bool(ok)); pick = d[0][1] if ok else pick
    return out, pick
res, recd, blocks, fl = [], [], {}, []
def run(Gv, y, sg, recrows, hsf, tag, sd):
    P = sg.copy(); hh = Gv.hour.values
    for r, ix in recrows.items():
        b = idx[r]
        if NUM[b] < 179: continue
        hs = hsf(r); vis = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in hs for i in range(N)])
        f, pick = fires(b, vis)
        used = 0
        for j in ix:
            if f[hh[j]]: P[j] = np.clip(sg[j] + LIFT, lo_, hi_); used += 1
        if used: fl.append(dict(val=tag, seed=str(sd), rec=r, hours=used, pick=rid(pick) if pick is not None else None, truly_high=bool(DM[b] >= 1.0)))
    p2 = Gv.day.values >= 179; d = (P - y) ** 2 - (sg - y) ** 2
    res.append(dict(val=tag, seed=str(sd), dSSE_all=float(d.sum()), dSSE_p2=float(d[p2].sum()), p2_pct=100 * (rm((P - y)[p2]) / rm((sg - y)[p2]) - 1)))
    bk = (Gv.farm + "_" + (Gv.day // 5).astype(str)).values
    blocks.setdefault((tag, "all"), []).append(pd.Series(d).groupby(bk).mean()); blocks.setdefault((tag, "p2"), []).append(pd.Series(d[p2]).groupby(bk[p2]).mean())
    for r_, v in pd.Series(d[p2]).groupby(Gv.rec.values[p2]).sum().items(): recd.append(dict(val=tag, seed=str(sd), rec=r_, dsse=v))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    run(Gv, y, sg, recrows, lambda r, hidden=hidden, fold_of=fold_of: hidden.get(fold_of[r], set()), v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}
    run(Gv, y, sg, recrows, lambda r, fo=fo: hid[fo[r]], "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); RD = pd.DataFrame(recd); FL = pd.DataFrame(fl); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
if len(FL): print("\nfired (seed-merged):", FL.groupby(["val", "rec"]).agg(hours=("hours", "mean"), pick=("pick", "first"), truly_high=("truly_high", "first")).to_string())
def boot(key):
    s = pd.concat(blocks[key], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s), (20000, len(s))); m = s[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot(("P2LOO", "p2")); harmed = RD[RD.val.isin(["P2LOO", "EL1"]) & (RD.dsse > 1e-6)]
c1b = pP2 < .025 or len(harmed) == 0
print(f"\n(1) main: cells better {int((main.dSSE_p2 < 0).sum())}/{len(main)} | P2LOO P(worse) {pP2:.4f} | harmed record-cells {len(harmed)} {harmed.groupby('rec').dsse.max().round(3).to_dict()} -> {'PASS' if c1 and c1b else 'FAIL'}")
ev = S[S.val == "EV"]; c2 = bool((ev.dSSE_all <= 0).all()); print(f"(2) EV: all-rows dSSE {ev.dSSE_all.round(3).tolist()} -> {'PASS' if c2 else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025); print(f"(3) DIAG10 guard: mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
g = RD.groupby(["val", "rec"]).dsse.mean(); print("pass-2 record changes:", g[g.abs() > 1e-6].round(3).to_dict())
print("EH1:", "ADOPT" if (c1 and c1b and c2 and c3) else "FAIL")
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; pm = P14.groupby("rec").sub_ec.mean()
visE = TRAIN & np.isfinite(DM); print("\neval records where EH1 fires (descriptive):")
for i in np.where(~TRAIN)[0]:
    if NUM[i] < 179: continue
    f, pick = fires(i, visE)
    if any(f): print(f"   {rid(i)}: hours {sum(f)}/24, ancestor {rid(pick) if pick is not None else None} (EC {DM[pick]:.2f}), 14th day-mean {pm.get(rid(i), np.nan):.3f} -> +{LIFT} on those hours")
