# -*- coding: utf-8 -*-
"""V12 (2026-10-13 집 클로드; 1단계 — 규정 안 짝찾기).  v11 과 같되 날짜를 DM11(평가 입력 사용) 대신 학습 전용 달력 DMT2(local/dmt2_days.csv: day, seg)로.
  대상 날짜 = 대상 0시 외기 4칸과 정확히 같은 '보이는 정답 기록'들의 DMT 날짜(최빈) + 그 seg; 후보 날짜 = 보이는 정답 기록이면 DMT, 아니면 같은 외기 일치 방식.
  cd1 = [같은 seg & 후보 날짜 = 대상 날짜 − 1], cdev = 같은 seg 일 때만 정의, 다르거나 모르면 cna = 1.
  가시성(규정): 대상 = 자기 0시 입력; 후보 = 학습 기록(번호·ID 무관) + 같은 ID 앞 번호 기록 입력(features10). 다른 ID 평가 입력 안 씀.
출력(로컬): v12_picks_mirror.csv(거울 CV o=0), v12_picks_ev.csv(EV 폴드 구조: 폴드의 val∪del 숨김), v12_picks_eval.csv(평가 60, 학습 전체).
  열: rec, prev, p, dT(대상 0시 실내 − 짝 23시 실내), date_status(ok / mismatch / unknown), labelled.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
H_ = os.path.join(env.ROOT, "집", "클로드", "research")
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
DT = pd.read_csv(os.path.join(env.LOCAL, "dmt2_days.csv")); DD_ = dict(zip(DT.rec, DT.day)); SG_ = dict(zip(DT.rec, DT.seg))
DAYT = np.array([DD_.get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([SG_.get(rid(i), np.nan) for i in range(N)])
F12 = ["oj"] + [f"v{j}" for j in range(8)] + ["fp", "excl", "cd1", "cdev", "cna"]
def date_of(i, lab):
    if lab[i] and np.isfinite(DAYT[i]): return DAYT[i], SEGT[i]
    vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[i, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return np.nan, np.nan
    d = pd.Series(DAYT[m]).mode().iloc[0]; s = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
    return d, s
def feats12(b, lab, removed):
    D, _ = features10(b, 0, lab, removed=removed)
    if len(D) == 0: return D
    a = D.a.values.astype(int)
    D["fp"] = np.nan_to_num(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0][None]), 4), axis=1), nan=1.0)
    db, sb = date_of(b, lab & (np.arange(N) != b)); dd = [date_of(x, lab) for x in a]
    da = np.array([x[0] for x in dd]); sa = np.array([x[1] for x in dd])
    ok = np.isfinite(da) & np.isfinite(db) & (sa == sb)
    D["cd1"] = np.where(ok, (da == db - 1).astype(float), 0.0); D["cdev"] = np.where(ok, np.minimum(np.abs(da - (db - 1)), 6) / 6, 0.0); D["cna"] = (~ok).astype(float)
    D["_dstat"] = np.where(ok, np.where(da == db - 1, "ok", "mismatch"), "unknown")
    return D
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats12(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df, y[0]))
def pick(b, lab, removed, w):
    D = feats12(b, lab, removed)
    if len(D) == 0: return None
    a = D.a.values.astype(int); sc = D[F12].values.astype(float) @ w; p = np.exp(sc - sc.max()); p /= p.sum(); k = int(np.argmax(p))
    return dict(rec=rid(b), prev=rid(a[k]), p=float(p[k]), dT=float(IN_[b, 0, 0] - IN_[a[k], 23, 0]), date_status=D._dstat.values[k], labelled=bool(lab[a[k]]))
fitw = lambda touch: fit_clogit([(d[F12].values.astype(float), y_) for b_, a_, d, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
OUT = os.path.join(env.LOCAL, "linker_v10")
# mirror o=0
rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fitw(touch); lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if TRAIN[b]:
                r = pick(b, lab, rem2, w)
                if r: rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v12_picks_mirror.csv"), index=False); print("mirror picks", len(rows), flush=True)
# EV geometry
FO = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); rows = []
for k, Gf in FO.groupby("fold"):
    val = [idx_ for idx_ in (np.where([rid(i) == r for i in range(N)])[0][0] for r in Gf.rec[Gf.role == "val"])]
    hidden = set(Gf.rec[Gf.role.isin(["val", "del"])]); hid_i = np.array([rid(i) in hidden for i in range(N)])
    dele = {i for i in range(N) if rid(i) in set(Gf.rec[Gf.role == "del"])}
    w = fitw(hid_i); lab = TRAIN & ~hid_i
    for b in val:
        r = pick(b, lab, dele, w)
        if r: r["fold"] = k; rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v12_picks_ev.csv"), index=False); print("EV picks", len(rows), flush=True)
# eval
w_all = fitw(np.zeros(N, bool)); rows = []
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    r = pick(b, TRAIN.copy(), set(), w_all)
    if r: rows.append(r)
E = pd.DataFrame(rows); E.to_csv(os.path.join(OUT, "v12_picks_eval.csv"), index=False)
print("weights:", dict(zip(F12, np.round(w_all, 2))))
print(E[(E.p >= .6) & E.labelled].to_string(index=False))
