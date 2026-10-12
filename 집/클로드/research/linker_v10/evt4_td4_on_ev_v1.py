# -*- coding: utf-8 -*-
"""EVT4 (2026-10-12 집 클로드; 자율 — TD4(6.743, DIAG10 계열 2차 −3.8%) 구성을 두 번째 독립 검증기 EV 에서 채점).  사전 고정(이 파일 커밋 후 실행).  온도만.
앵커 집합 = TD4: v7 거울 CV 앵커(p ≥ .8·정답·|dT| ≤ 1.5) ∪ v11 거울 CV 짝(p ≥ .6·정답, v7 없는 기록; |dT| > 1.5 여도 사용 = 차단 해제) − 날짜 불일치(대상 날짜 = 거울 덩어리에서 보이는 정답 기록과 0시 외기 일치, 짝 날짜 DM11, 차 ≠ 1).
모두 λ1, 13회차 식(q18_h + BETA_h·clip(A − .36 − q0, ±2)), 전날이 EV 폴드 val/del 이면 버림. 기준 = 저장 v22(13회차 구성).
판정(단일 안 α .025): 4칸 모두 전체 개선 AND 2차 개선 AND 기록 부트스트랩 P(악화) < .025 AND 상위 1기록 몫 ≤ .5.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
H_ = os.path.join(env.ROOT, "집", "클로드", "research")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
def mirror_lab(b):
    f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
def tdate(b):
    lab = mirror_lab(b); vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
A = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
P11 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv")); P11 = P11[P11.p >= .6]
nadd = ndrop = 0
for r, p in zip(P11.rec, P11.prev):
    if r in A: continue
    g = tdate(idx[r]) - DAY[idx[p]]
    if np.isfinite(g) and g != 1: ndrop += 1; continue
    A[r] = p; nadd += 1
print(f"anchors: v7 {len(A) - nadd} + v11 added {nadd} (date-mismatch dropped {ndrop}) = {len(A)}")
FO = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
DD = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rng = np.random.default_rng(20261025); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = []
for c in ["7A", "7B", "101A", "101B"]:
    q18 = DD[f"v18_a_{c}"].values; base = DD[f"v22_a_{c}"].values; y = DD.y.values; p2 = DD.day.values >= 179; new = q18.copy()
    for rec, G in DD.groupby("rec"):
        if rec not in A or A[rec] in hid[int(G.fold.iloc[0])]: continue
        ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
        new[ix] = q18[ix] + BETA[hh] * np.clip(ST[idx[A[rec]], 23] - .36 - q0, -2, 2)
    db = pd.Series((base - y) ** 2).groupby(DD.rec.values).sum(); dn = pd.Series((new - y) ** 2).groupby(DD.rec.values).sum(); dl = (dn - db).values
    pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
    res.append(dict(cell=c, all_pct=100 * (rm(new - y) / rm(base - y) - 1), p2_pct=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1), P_worse=pw,
                    top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan, changed=int((np.abs(dl) > 1e-12).sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 200); print(S.round(4).to_string(index=False))
ok = (S.all_pct < 0).all() and (S.p2_pct < 0).all() and (S.P_worse < .025).all() and S.top_share.mean() <= .5
print("EVT4:", "ADOPT" if ok else "FAIL")
