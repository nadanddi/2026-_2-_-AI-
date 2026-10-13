# -*- coding: utf-8 -*-
"""HL1 서술 점검 (2026-10-13 집 클로드; 사용자: '고EC 기록 짝 찾기(ID 넘나듦 포함) 규정 안에서 해봐').
규칙(규정 안): 대상 b 의 날짜 = 0시 외기 4칸이 같은 보이는 정답 기록들의 학습 전용 달력 DMT3e 날짜 최빈값 d·구간 s.
  후보 = 보이는 정답(학습) 기록 중 하루 평균 EC ≥ 1.0, 같은 구간 날짜 d−1 (ID 무관) & 그 후보의 '확실 연결 다음날'이 이미 보이는 정답 기록이면 제외.
  선택 = 0..h 운영 지문(학습 기록으로만 표준화) 최근접; 후보 1개면 그것, 2개 이상이면 2등/1등 거리 ≥ 1.5 일 때만.
점검: 학습 고EC 기록(하루 평균 ≥ 1.0, 깨끗한 진짜 전날 있음)에서 대상 자신만 숨기고(가장 너그러운 기하) 고른 짝이 진짜 전날인 비율·발동률,
  그리고 h = 0, 9, 23 별.  평가 60 중 모델이 높게 본 기록(12회차 하루 평균 ≥ 1.0)에 대한 선택.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {int(b): int(a) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
def date_of(b, lab):
    vi = np.where(lab)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return None
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
def hl1(b, lab, h):
    td = date_of(b, lab)
    if td is None: return None, "nodate", 0
    d, s = td; c = [a for a in np.where(lab)[0] if a != b and DAYT[a] == d - 1 and SEGT[a] == s and DM[a] >= 1.0 and not any(lab[x] and DAYT[x] == d for x in SUCC.get(a, []))]
    if not c: return None, "nocand", 0
    dist = sorted((float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4))), a) for a in c)
    if len(dist) == 1 or dist[1][0] >= 1.5 * dist[0][0]: return dist[0][1], "pick", len(c)
    return None, "ambig", len(c)
rows = []
for b in np.where(TRAIN & (DM >= 1.0))[0]:
    if b not in CLEAN: continue
    lab = TRAIN.copy(); lab[b] = False
    for h in (0, 9, 23):
        a, st, nc = hl1(b, lab, h); rows.append(dict(rec=rid(b), h=h, status=st, ncand=nc, ok=(a == CLEAN[b]) if a is not None else None, true_prev=rid(CLEAN[b]), true_prev_high=DM[CLEAN[b]] >= 1.0, cross_id=FARM[CLEAN[b]] != FARM[b], p2=NUM[b] >= 179))
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(f"train high records with clean prev: {R.rec.nunique()} | true prev itself high: {R.drop_duplicates('rec').true_prev_high.mean():.2f} | cross-ID: {R.drop_duplicates('rec').cross_id.mean():.2f}")
print(R.groupby("h").apply(lambda g: pd.Series(dict(fire=(g.status == "pick").mean(), acc=g.ok.dropna().astype(bool).mean() if g.ok.notna().any() else np.nan, n_pick=int(g.ok.notna().sum()), nodate=(g.status == "nodate").mean(), nocand=(g.status == "nocand").mean(), ambig=(g.status == "ambig").mean()))).round(3).to_string())
print(R[R.h == 23][["rec", "status", "ncand", "ok", "true_prev", "cross_id", "p2"]].to_string(index=False))
P12 = pd.read_csv(os.path.join(env.ROOT, "제출", "12회차_2026-10-11(나)", "submission_19.csv")); P12["rec"] = P12.row_id.str[:7]; pm = P12.groupby("rec").sub_ec.mean()
print("\neval records predicted high (12th >= 1.0):")
for r in pm[pm >= 1.0].index:
    b = idx[r]; out = []
    for h in (0, 9, 23):
        a, st, nc = hl1(b, TRAIN.copy(), h); out.append(f"h{h}: {st} {rid(a) if a is not None else ''} {'' if a is None else f'(23h {EC[a,23]:.2f})'} n{nc}")
    print(f"  {r} pred {pm[r]:.2f} | " + " | ".join(out))
# --- model-high records (DIAG10 OOF sg day mean >= 1.0): HL1 status vs actual level (target only hidden) ---
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
sg = O.groupby(O.row_id.str[:7]).sg.mean()
rows2 = []
for r, v in sg[sg >= 1.0].items():
    b = idx[r]; lab = TRAIN.copy(); lab[b] = False; a, st, nc = hl1(b, lab, 23)
    rows2.append(dict(rec=r, sg=v, y=DM[b], status=st, high=DM[b] >= 1.0))
T2 = pd.DataFrame(rows2)
print("\nmodel-high train records by HL1 status (h23): n, share truly high (>=1.0), mean y, mean sg")
print(T2.groupby("status").agg(n=("rec", "size"), truly_high=("high", "mean"), y=("y", "mean"), sg=("sg", "mean")).round(3).to_string())
print(T2[T2.status != "pick"].sort_values("sg", ascending=False).round(3).to_string(index=False))
