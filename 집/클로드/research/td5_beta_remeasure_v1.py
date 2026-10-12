# -*- coding: utf-8 -*-
"""TD5 (2026-10-12 집 클로드; 사용자 부재 중 자율 실험 — 근거: 현 BETA(ta8)는 v7 앵커 1차 32기록으로 잰 값, TD4 앵커 집합은 1차가 훨씬 많음).  사전 고정(이 파일 커밋 후 실행).
안 = TD4 앵커 집합 그대로, BETA 만 재측정: DIAG10 4칸 각각에서 1차(번호 < 179) TD4 앵커 기록으로 시각별 최소제곱 기울기 β_h = Σ gap·(y_h − q_h) / Σ gap²,
  gap = A − .36 − q0 (자르기 전), 4칸 평균 → BETA_new. 판정: TD4(현 BETA) 대비 DIAG10 2차 4칸 모두 개선 AND 2차 기록 부트스트랩 P(악화) < .025 AND 잭나이프-1 같은 방향. 다른 검증기 서술.
(아래 TD4 원문)
TD4 (2026-10-12 집 클로드; 사용자: '불일치 짝만 빼는 방식') = TD2 와 같은 규칙·틀·판정 + 날짜 불일치 짝 제외:
  v11 짝(원래 확신 유지)의 대상 날짜 = 그 거울 덩어리에서 보이는 정답 기록과 0시 외기 4칸 정확 일치(DM11 최빈), 짝 날짜 = DM11; 둘 다 알고 차 ≠ 1 이면 그 v11 짝 제외(교체 없음). v7 앵커는 그대로.
(아래 TD2 원문)
TD2 (2026-10-12 집 클로드; 사용자: '그 가설로 검증' = 6.740: 확신 높은데 실내 차로 차단된 짝은 살리고, 확신 낮은 짝은 버림).  사전 고정(이 파일 커밋 후 실행).  온도만.
안 CERT2 = TD1 의 CERT ∪ v11 확신 p ≥ .6 & |dT_air| > 1.5 인 기록(차단 해제, λ1; v7 앵커 있는 기록은 v7 그대로). 판정·틀은 TD1 과 동일. TD1(CERT) 대비 변화도 서술.
(아래는 TD1 원문)
EV11c(서술)에서 애매 층(λ .5)은 순효과 ≈ 0 → '확실 층만' 안을 EV 가 아닌 DIAG10 계열(TA8 판정과 같은 틀)에서 판정.
틀: ta7/ta2 A1.frames() — 11회차 구성 OOF(q), 앵커 공식 = 13회차 제출 규칙(ta7.correct: p ≥ .8, 편향 상수 .36, 측정 감쇠 BETA(ta8_beta), 자르기 2, 실내 차 1.5).
  기준 BASE = v7 거울 CV 선택(cv_all400_choice_v7m, 정답·1단계) 로 앵커 = 13회차 구성.
  안 CERT = BASE 앵커 ∪ v11 거울 CV 선택(local/linker_v10/v11_mirror_picks.csv) 중 확신 p ≥ .6 & |dT_air| ≤ 1.5 인 기록(v7 앵커가 있는 기록은 v7 그대로).
판정(단일 안, α .025): DIAG10 2차 행 RMSE 4칸 모두 개선 AND 2차 기록 단위 부트스트랩(20000, 4칸 평균 ΔSSE) P(악화) < .025 AND 최대 이득 1기록 제외(잭나이프-1)해도 같은 방향.
서술: DIAG10 전체, G, EL1, EXT10, EXT12 의 2차·전체 변화, 추가 앵커 수.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import ST, FARM, NUM, N  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
L7 = pd.read_csv(A7.LK); L7 = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1)].copy()
v7set = set(L7[(L7.p >= .8) & (L7.dT_air.abs() <= 1.5)].rec)
P11 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv"))
P11 = P11[(P11.p >= .6) & ~P11.rec.isin(v7set)]
from feat4 import X as OUTX, TRAIN as TRN  # noqa: E402
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
def mirror_lab(b):
    f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRN & ~touch
def tdate(b):
    lab = mirror_lab(b); vi = np.where(lab)[0]; eq = np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
gap = np.array([tdate(idx[r]) - DAY[idx[p]] for r, p in zip(P11.rec, P11.prev)])
mism = np.isfinite(gap) & (gap != 1)
print(f"v11 picks p>=.6 not in v7: {len(P11)} | date-mismatched dropped {int(mism.sum())}: {sorted(P11.rec[mism].tolist())}")
P11 = P11[~mism]
gated = P11.dT.abs() > 1.5
add = pd.DataFrame(dict(rec=P11.rec, prev=P11.prev, p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in P11.prev], dT_air=np.where(gated, 0.0, P11.dT), steps_to_anchor=1))
LT1 = pd.concat([L7[L7.rec.isin(v7set)], add[~gated.values]], ignore_index=True)
LC = pd.concat([L7[L7.rec.isin(v7set)], add], ignore_index=True)
print(f"anchors: BASE {len(v7set)} | TD1 CERT {len(LT1)} | TD2 CERT2 {len(LC)} (gate-released confident {int(gated.sum())}: {sorted(P11.rec[gated].tolist())})")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
R = A7.A2.A1.frames(); rng = np.random.default_rng(20261021)
M, cells = R["DIAG10"]; kc, Ac = A7.build(M, LC); h = M.hour.values; y = M.sub_temp.values; p1 = M.day.values < 179
sel = Ac[(Ac.p >= .8) & (Ac.dT_air.abs() <= 1.5)]
num = np.zeros(24); den = np.zeros(24)
for q in cells:
    q0 = pd.Series(q[h == 0], index=kc[h == 0])
    for rec in sel.index.intersection(q0.index):
        if int(rec[4:7]) >= 179: continue
        g = sel.at[rec, "anchor_st23"] - .36 - q0[rec]; m = kc == rec
        num[h[m]] += g * (y[m] - q[m]); den[h[m]] += g * g
BETA_OLD = A7.BETA.copy(); BETA_NEW = num / den
print("pass-1 anchored records used:", int(sum(1 for r in sel.index if int(r[4:7]) < 179)))
print("BETA old:", " ".join(f"{x:.2f}" for x in BETA_OLD)); print("BETA new:", " ".join(f"{x:.2f}" for x in BETA_NEW))
out = []
for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
    if Lname not in R: continue
    M, cells = R[Lname]; y = M.sub_temp.values; p2 = M.day.values >= 179; kc, Ac = A7.build(M, LC); d_all, d_p2, dsse = [], [], []
    for q in cells:
        A7.BETA = BETA_OLD; b = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
        A7.BETA = BETA_NEW; c = A7.correct(M, q, kc, Ac, .8, "const", "meas", 2, 1.5)
        d_all.append(100 * (rm(c - y) / rm(b - y) - 1)); d_p2.append(100 * (rm((c - y)[p2]) / rm((b - y)[p2]) - 1))
        dsse.append(pd.Series(((c - y) ** 2 - (b - y) ** 2)[p2]).groupby(kc[p2]).sum())
    D = pd.concat(dsse, axis=1).mean(axis=1)
    out.append(dict(layout=Lname, all_pct=" ".join(f"{x:+.2f}" for x in d_all), p2_pct=" ".join(f"{x:+.2f}" for x in d_p2)))
    if Lname == "DIAG10":
        v = D.values; P = float((v[rng.integers(0, len(v), (20000, len(v)))].sum(1) >= 0).mean()); jk = D.drop(D.idxmin()).sum()
        ok = all(x < 0 for x in d_p2) and P < .025 and jk < 0
        print(f"DIAG10 pass-2 (new vs old BETA): {[round(x, 2) for x in d_p2]} | P {P:.4f} | jackknife {jk:+.3f} -> {'ADOPT' if ok else 'FAIL'}")
A7.BETA = BETA_OLD
pd.set_option("display.width", 200); print(pd.DataFrame(out).to_string(index=False))
np.save(os.path.join(env.LOCAL, "linker_v10", "td5_beta_new.npy"), BETA_NEW)
