# -*- coding: utf-8 -*-
"""HF1 (2026-10-13 집 클로드; 사용자: '있는데 못 찾은 것부터 해결') 서술(신호 확인).
질문: v15(PI .15) 1순위 짝이 맞는지를 대상의 0..h시 운영 지문(8변수, 규정: h시 행은 자기 0..h시 입력)으로 더 확신할 수 있나.
지문 거리 d_h(후보) = mean_{t≤h, 변수} min(|FPA[대상,t] − FPA[후보,t]|, 4)  (같은 동은 날마다 운영이 비슷하다는 KD1 6.639 근거).
v15 상위 6 실제 후보(보이는 정답 기록만) 중 1순위의 d_h 순위·2등과의 거리 차. 정답(깨끗한 정답) 기준 1순위 맞음 vs 틀림, 그리고 '있는데 못 찾음'(p0<.8 & 맞음).
h = 0, 3, 6, 9, 12, 18, 23."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
from sklearn.metrics import roc_auc_score
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(os.path.join(S3, "truth.pkl")); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
clean = lambda r: r in TP and mirror_lab(r)[idx[TP[r]]] and DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) == 1
D = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_mirror.csv"))
dh = lambda b, a, h: float(np.nanmean(np.minimum(np.abs(FPA[a, :h + 1] - FPA[b, :h + 1]), 4)))
rows = []
for _, r in D.iterrows():
    if not r.lab0 or not clean(r.rec): continue
    b = idx[r.rec]; cands = [r[f"prev{k}"] for k in range(6) if isinstance(r.get(f"prev{k}"), str) and bool(r[f"lab{k}"])]
    if len(cands) < 2: continue
    row = dict(rec=r.rec, pass2=int(r.rec[4:7]) >= 179, p0=r.p0, ok=r.prev0 == TP[r.rec], ncand=len(cands))
    for h in (0, 3, 6, 9, 12, 18, 23):
        d = {c: dh(b, idx[c], h) for c in cands}; o = sorted(d, key=d.get)
        row[f"rank{h}"] = o.index(r.prev0) + 1; row[f"gap{h}"] = (min(v for c, v in d.items() if c != r.prev0) - d[r.prev0])
        row[f"truth_rank{h}"] = o.index(TP[r.rec]) + 1 if TP[r.rec] in d else np.nan
    rows.append(row)
T = pd.DataFrame(rows); T["missed"] = T.ok & (T.p0 < .8)
print(f"targets with clean truth & >=2 visible labelled candidates: {len(T)} | top-1 correct {T.ok.sum()} | correct but p0<.8 (missed) {T.missed.sum()} (pass-2 {T[T.pass2].missed.sum()})")
for h in (0, 3, 6, 9, 12, 18, 23):
    print(f"h={h:2d}: top-1 is fingerprint-nearest: correct {np.mean(T[T.ok][f'rank{h}'] == 1):.2f} | wrong {np.mean(T[~T.ok][f'rank{h}'] == 1):.2f} | missed {np.mean(T[T.missed][f'rank{h}'] == 1):.2f}"
          f" | AUC(gap) correct-vs-wrong {roc_auc_score(T.ok, T[f'gap{h}']):.3f} | truth fingerprint-nearest (all) {np.mean(T[f'truth_rank{h}'] == 1):.2f}")
pd.set_option("display.width", 220)
print("\nmissed (correct top-1, p0<.8) pass-2:"); print(T[T.missed & T.pass2][["rec", "p0", "ncand"] + [f"rank{h}" for h in (0, 6, 12, 23)] + [f"gap{h}" for h in (0, 6, 12)]].round(3).to_string(index=False))
print("\nwrong top-1:"); print(T[~T.ok][["rec", "pass2", "p0", "ncand"] + [f"rank{h}" for h in (0, 6, 12, 23)] + [f"truth_rank{h}" for h in (0, 6, 12)]].round(3).to_string(index=False))
