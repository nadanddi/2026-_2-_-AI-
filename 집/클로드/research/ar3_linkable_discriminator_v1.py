# -*- coding: utf-8 -*-
"""AR3 (2026-10-13 집 클로드; 사용자: '맞춘 개수가 줄어도 구분력을 최대로 — 그래야 다른 분류도').  서술(신호 확인).
정답: 짝 상태 맞음(깨끗한 정답 = 진단 연결·달력 차 1 이고 짝이 그것) = 1 / 그 외(틀림·확인 불가) = 0.
표본: 거울 CV 기하의 v7 짝(정답 기록, 확신 무관) ∪ v12 짝(정답 기록, 확신 무관), (rec, prev) 중복 제거.
특징(평가 때 알 수 있는 것만; 날짜는 학습 전용 달력 DMT2, 보이는 정답 기록 = 거울 숨김 반영):
  링커: p, by_v7, by_v12, both, hp2_agree, date_ok, date_unknown, day_gap(짝 날짜 − (대상 날짜−1), 같은 구간일 때만)
  날짜 결정력: n_exact(대상 0시 외기와 똑같은 보이는 정답 기록 수), n_dm1(대상 날짜−1·같은 구간의 보이는 정답 기록 수; 0 이면 '전날 정답 없음')
  짝 점유: prev_taken(짝의 다음날로 이미 보이는 정답 기록이 확실 연결됨)
  연속성: oj(외기 자정 연속 비용, v7 식), 실내 3변수 점프(그대로 + 기울기 외삽), 운영 장치 7개 점프 평균, fp(0시 지문 거리), fp_rank(전날 후보 중 지문 순위)
  EC: gapA(|짝 23시 EC − 대상 0시 예측|).  다른 ID(cross_id)는 넣은 판/뺀 판 둘 다.
모델: 로지스틱(표준화) vs HistGradientBoosting, 거울 블록 단위 교차적합.  보고: AUC, '믿음' 순으로 덮을 때 포괄 30/50/70% 지점의 정확도, 확신 ≥ .6 부분의 AUC,
  그리고 손해 큰 짝(F47_146 등)·이득 큰 짝(F47_132 등)의 순위.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])  # hp2, idx, rid, DAYT, SEGT, SUCC, FPA, OUTX, IN_, TRAIN, FARM, NUM, N, S3
from feat4 import EC, AC, sdO  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
blk = lambda r: f"{r[:3]}_{(int(r[4:7]) - NUM[FARM == r[:3]].min()) // 10}"
TL = pd.read_pickle(S3 + r"\truth.pkl"); TPd = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))  # 정답 표시용(학습 기록 날짜)만
def clean_ok(r, pv):
    if r not in TPd or not mirror_lab(r)[idx[TPd[r]]] or DAYr.get(r, np.nan) - DAYr.get(TPd[r], np.nan) != 1: return 0
    return int(TPd[r] == pv)
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
v7p = {r: (pv, p) for r, pv, p, l in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled) if l and isinstance(pv, str)}
v12p = {r: (pv, p, ds) for r, pv, p, ds, l in zip(PM.rec, PM.prev, PM.p, PM.date_status, PM.labelled) if l}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]
SG0 = O.assign(rec=O.row_id.str[:7]).groupby("rec").sg.mean().to_dict()
W = np.array([1, 1, .3, .5]); oj = lambda a, b: float(np.sum(W * np.abs(OUTX[a, 23] + .5 * (OUTX[a, 23] - OUTX[a, 22]) - OUTX[b, 0]) / sdO))
INsd = np.nanstd(IN_.reshape(-1, IN_.shape[2]), 0) + 1e-9; ACsd = np.nanstd(AC.reshape(-1, AC.shape[2]), 0) + 1e-9
def tdate(b, lab):
    vi = np.where(lab)[0]; eq = np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1); m = vi[eq]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return np.nan, np.nan, int(eq.sum())
    d = pd.Series(DAYT[m]).mode().iloc[0]; s = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
    return d, s, int(eq.sum())
rows = []
for r in sorted(set(v7p) | set(v12p)):
    if r not in SG0: continue
    b = idx[r]; lab = mirror_lab(r); lab[b] = False; d, s, nex = tdate(b, lab)
    cand = np.where(lab & (DAYT == d - 1) & (SEGT == s))[0] if np.isfinite(d) else np.array([], int)
    fpd = {a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4))) for a in cand}
    h = hp2(b, mirror_lab(r))
    for pv in {x[0] for x in (v7p.get(r), v12p.get(r)) if x}:
        a = idx[pv]
        if not np.isfinite(EC[a, 23]): continue
        pa7 = v7p[r][1] if r in v7p and v7p[r][0] == pv else 0.0; pa12 = v12p[r][1] if r in v12p and v12p[r][0] == pv else 0.0
        ds = v12p[r][2] if r in v12p and v12p[r][0] == pv else "na"
        same = np.isfinite(d) and np.isfinite(DAYT[a]) and SEGT[a] == s
        f_a = float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
        rows.append(dict(rec=r, prev=pv, blk=blk(r), pass2=int(r[4:7]) >= 179, y=clean_ok(r, pv),
            p=max(pa7, pa12), p7=pa7, p12=pa12, by_v7=float(pa7 > 0), by_v12=float(pa12 > 0), both=float(pa7 > 0 and pa12 > 0),
            hp2_agree=float(h == pv), hp2_none=float(h is None), date_ok=float(ds == "ok"), date_unknown=float(ds in ("unknown", "na")),
            day_gap=float(min(abs(DAYT[a] - (d - 1)), 6)) if same else 7.0, n_exact=nex, n_dm1=len(cand),
            prev_taken=float(any(lab[x] and np.isfinite(d) and DAYT[x] == d for x in SUCC.get(a, []))),
            oj=oj(a, b), cross_id=float(r[:3] != pv[:3]),
            jT=abs(IN_[b, 0, 0] - IN_[a, 23, 0]) / INsd[0], jH=abs(IN_[b, 0, 1] - IN_[a, 23, 1]) / INsd[1], jC=abs(IN_[b, 0, 2] - IN_[a, 23, 2]) / INsd[2],
            xT=abs(IN_[a, 23, 0] + .5 * (IN_[a, 23, 0] - IN_[a, 22, 0]) - IN_[b, 0, 0]) / INsd[0],
            xC=abs(IN_[a, 23, 2] + .5 * (IN_[a, 23, 2] - IN_[a, 22, 2]) - IN_[b, 0, 2]) / INsd[2],
            jAct=float(np.nanmean(np.abs(AC[a, 23] - AC[b, 0]) / ACsd)), fp=f_a,
            fp_rank=float(1 + sum(v < f_a - 1e-12 for v in fpd.values())) if a in fpd else 9.0,
            gapA=abs(EC[a, 23] - SG0[r]), d0=(EC[a, 23] - EC[b, 0]) ** 2 - (SG0[r] - EC[b, 0]) ** 2))
T = pd.DataFrame(rows); T = T.fillna(T.median(numeric_only=True))
print(f"pairs {len(T)} | linkable(y=1) {T.y.sum()} | pass-2 {T.pass2.sum()} (y=1 {T[T.pass2].y.sum()}) | p>=.6 {int((T.p >= .6).sum())}", flush=True)
FEAT = ["p", "p7", "p12", "by_v7", "by_v12", "both", "hp2_agree", "hp2_none", "date_ok", "date_unknown", "day_gap", "n_exact", "n_dm1", "prev_taken",
        "oj", "jT", "jH", "jC", "xT", "xC", "jAct", "fp", "fp_rank", "gapA", "cross_id"]
print("\nsingle-feature AUC (y=1 linkable; <.5 means larger -> less linkable):")
print({f: round(roc_auc_score(T.y, T[f]), 3) for f in FEAT})
def cv(F, kind):
    oof = np.zeros(len(T)); Z = T[F].values.astype(float)
    for g in T.blk.unique():
        tr, te = (T.blk != g).values, (T.blk == g).values
        if kind == "logit":
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
            oof[te] = LogisticRegression(C=.3, max_iter=4000).fit((Z[tr] - mu) / sd, T.y[tr]).predict_proba((Z[te] - mu) / sd)[:, 1]
        else:
            oof[te] = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=10, l2_regularization=1.0,
                                                     random_state=0).fit(Z[tr], T.y[tr]).predict_proba(Z[te])[:, 1]
    return oof
def report(nm, s):
    o = np.argsort(-s); yy = T.y.values[o]; out = []
    for c in (.3, .5, .7):
        k = int(round(c * T.y.sum())); kk = np.searchsorted(np.cumsum(yy), k) + 1; out.append(f"recall {c:.0%}: precision {k / kk:.3f} (kept {kk})")
    hi = T.p.values >= .6; p2 = T.pass2.values
    print(f"{nm:34s} AUC {roc_auc_score(T.y, s):.3f} | p>=.6 AUC {roc_auc_score(T.y[hi], s[hi]):.3f} | pass-2 AUC {roc_auc_score(T.y[p2], s[p2]):.3f} | " + " | ".join(out))
print()
report("baseline: linker p", T.p.values)
FNC = [f for f in FEAT if f != "cross_id"]
S = {}
for nm, F in (("logit all", FEAT), ("logit no cross_id", FNC), ("gbm all", FEAT), ("gbm no cross_id", FNC)):
    S[nm] = cv(F, nm.split()[0]); report(nm, S[nm])
S["blend no cross_id"] = (pd.Series(S["logit no cross_id"]).rank().values + pd.Series(S["gbm no cross_id"]).rank().values) / (2 * len(T)); report("blend(logit+gbm) no cross_id", S["blend no cross_id"])
T["score"] = S["blend no cross_id"]; T["rank_trust"] = T.score.rank(ascending=False)
pd.set_option("display.width", 230)
print("\nkey pairs (rank_trust 1 = most trusted, of", len(T), "):")
key = ["F47_146", "F13_177", "F13_116", "F13_214", "F13_149", "F47_079", "F47_132", "F13_217", "F47_178", "F13_241"]
print(T[T.rec.isin(key)].sort_values("rank_trust")[["rec", "prev", "y", "p", "n_dm1", "prev_taken", "day_gap", "hp2_agree", "fp_rank", "jC", "gapA", "d0", "score", "rank_trust"]].round(3).to_string(index=False))
m = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=10, l2_regularization=1.0, random_state=0).fit(T[FNC].values, T.y)
from sklearn.inspection import permutation_importance
pi = permutation_importance(m, T[FNC].values, T.y, scoring="roc_auc", n_repeats=10, random_state=0)
print("\ngbm permutation importance (in-sample, top 10):", dict(sorted(zip(FNC, np.round(pi.importances_mean, 3)), key=lambda x: -x[1])[:10]))
T.to_csv(os.path.join(env.LOCAL, "ar3_pairs.csv"), index=False)
