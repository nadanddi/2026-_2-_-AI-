# -*- coding: utf-8 -*-
"""LKH1 (2026-10-14 집 클로드; 사용자: '시각별 갱신판 한 번 만들어봐').  서술·탐색(판정 아님).
시각별 갱신 링커: 대상 b 의 h 시 예측용 짝 = 학습 달력(DMT3e) 날짜가 '대상 날짜 − 1'(같은 구간)인 후보 중, 운영 지문 0..h 시가 가장 가까운 기록.
  대상 날짜 = 0시 외기 일치 정답 기록들의 최빈 날짜(대상 제외).  지문 = 구동기·CO₂ 8열(HL1 과 같음, 학습 기록으로 표준화), 거리 = 0..h 시 평균 |차|(4 로 자름).
  규정: 대상은 0..h 만, 후보는 앞 기록(정답 있는 학습 기록)이라 같은 시각대 비교.
평가(학습): cert 연결 쌍에서 대상만 숨기고, 진짜 전날이 후보에 있는 쌍만 — 시각별 1순위 정확도, 2등/1등 거리비 ≥ 1.5(확신) 일 때 정확도·비율.
  기준: 0시 지문만(h=0), v15R 거울 1순위(p0) 정확도(같은 쌍).
평가 기록 서술: F13_221 의 시각별 1순위 후보(후보 = 날짜 11·구간 1 의 정답 기록 + 같은 ID 앞 번호 평가 기록).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def tdate(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return pd.Series(ds).mode().iloc[0] if ds else None
def dist(a, b, h): return float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); P0 = dict(zip(PM.rec, PM.prev0))
HS = [0, 3, 6, 8, 9, 10, 12, 15, 18, 23]; rows = []
for bb, aa in CERT.items():
    b = idx[bb]; lab = TRAIN.copy(); lab[b] = False; td = tdate(b, lab)
    if td is None: continue
    cands = [i for i in np.where(lab)[0] if DAYT.get(rid(i)) == td[0] - 1 and SEGT.get(rid(i)) == td[1]]
    if idx[aa] not in cands: continue
    out = dict(rec=bb, ncand=len(cands), v15_ok=P0.get(bb) == aa)
    for h in HS:
        d = sorted((dist(a, b, h), a) for a in cands); top = d[0][1]; conf = len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)
        out[f"ok{h}"] = top == idx[aa]; out[f"conf{h}"] = conf
    rows.append(out)
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"cert pairs with date found and true prev among date-1 candidates: {len(R)} (mean candidates {R.ncand.mean():.1f}) | v15R mirror top-1 acc on same pairs {R.v15_ok.mean():.3f}")
print("hour | top-1 acc | confident share | acc when confident")
for h in HS: print(f"{h:4d} | {R[f'ok{h}'].mean():.3f} | {R[f'conf{h}'].mean():.3f} | {R[R[f'conf{h}']][f'ok{h}'].mean():.3f}")
print("pairs where v15R wrong: acc by hour", {h: round(float(R[~R.v15_ok][f'ok{h}'].mean()), 3) for h in HS}, f"(n {int((~R.v15_ok).sum())})")
# eval descriptive: F13_221
b = idx["F13_221"]; td = tdate(b, TRAIN.copy()); print("\nF13_221 training-calendar date:", td)
cands = [i for i in range(N) if DAYT.get(rid(i)) == td[0] - 1 and SEGT.get(rid(i)) == td[1] and TRAIN[i]] + [idx[r] for r in ("F13_220",) if False]
for h in HS:
    d = sorted((dist(a, b, h), rid(a)) for a in cands); print(f"  h{h:2d}: " + ", ".join(f"{r} {x:.2f}" for x, r in d[:4]))
print("  (same-fingerprint house F13_123, date 7) distance by hour:", {h: round(dist(idx['F13_123'], b, h), 2) for h in HS})
R.to_csv(os.path.join(HERE, "lkh1_pairs_v1.csv"), index=False)
