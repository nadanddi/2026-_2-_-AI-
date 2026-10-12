# -*- coding: utf-8 -*-
"""AR2 (2026-10-13 집 클로드; 사용자: '전날 있는지 없는지 잘 구분한 애들과 못한 애들 추려봐') 서술.
정답 = 짝 상태: 맞음(깨끗한 정답 = 진단 연결·달력 차 1 이고 짝이 그것) / 틀림·확인 불가(그 외).  AR1 짝 119·같은 특징·거울 블록 교차적합 로지스틱 → 의심 점수.
의심 상위 25% 를 '없음(의심)' 으로 판정. 네 칸: 맞음&안심(잘 구분) / 틀림&의심(잘 구분) / 틀림&안심(놓침) / 맞음&의심(헛경보). 칸별 0시 손익·특징 평균·기록 목록."""
import os, sys, env, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "linker_v4")); from feat4 import EC, FARM, NUM, N, S3, TRAIN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T = pd.read_csv(os.path.join(env.LOCAL, "ar1_pairs.csv")); rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
TL = pd.read_pickle(S3 + r"\truth.pkl"); TPd = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
CAL = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))
def status(r, pv):
    if r not in TPd or not mirror_lab(r)[idx[TPd[r]]] or DAYr.get(r, np.nan) - DAYr.get(TPd[r], np.nan) != 1: return "unverifiable"
    return "correct" if TPd[r] == pv else "wrong"
T["link"] = [status(r, p) for r, p in zip(T.rec, T.prev)]; T["nolink"] = (T.link != "correct").astype(int)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]
s0 = O.assign(rec=O.row_id.str[:7]).groupby("rec").sg.mean().reindex(T.rec).values
A = np.array([EC[idx[p], 23] for p in T.prev]); y0 = np.array([EC[idx[r], 0] for r in T.rec]); T["d0"] = (A - y0) ** 2 - (s0 - y0) ** 2
FEAT = ["p", "by_v7", "by_v12", "both", "hp2_agree", "date_ok", "cross_id", "jT", "jH", "jC", "fp", "gapA"]
oof = np.zeros(len(T))
for g in T.blk.unique():
    tr, te = (T.blk != g).values, (T.blk == g).values; Z = T[FEAT].values.astype(float); mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    oof[te] = LogisticRegression(C=.5, max_iter=2000).fit((Z[tr] - mu) / sd, T.nolink[tr]).predict_proba((Z[te] - mu) / sd)[:, 1]
T["susp"] = oof; flag = oof > np.quantile(oof, .75)
T["cell"] = np.select([(T.nolink == 0) & ~flag, (T.nolink == 1) & flag, (T.nolink == 1) & ~flag, (T.nolink == 0) & flag],
                      ["1 good-link & trusted (OK)", "2 no-link & flagged (OK)", "3 no-link & trusted (MISSED)", "4 good-link & flagged (FALSE ALARM)"], default="")
print(f"pairs {len(T)} | link status {T.link.value_counts().to_dict()} | block-CV AUC {roc_auc_score(T.nolink, oof):.3f}")
pd.set_option("display.width", 230)
print(T.groupby("cell").agg(n=("rec", "size"), pass2=("pass2", "sum"), d0_sum=("d0", "sum"), p=("p", "mean"), hp2=("hp2_agree", "mean"), cross=("cross_id", "mean"),
                            jT=("jT", "mean"), jC=("jC", "mean"), fp=("fp", "mean"), gapA=("gapA", "mean")).round(3).to_string())
C = ["rec", "prev", "link", "pass2", "susp", "d0", "p", "hp2_agree", "cross_id", "jT", "jC", "gapA"]
for c in sorted(T.cell.unique()):
    S = T[T.cell == c].sort_values("susp", ascending=c.startswith(("1", "3")))
    print(f"\n== {c} ({len(S)})"); print(S[C].round(3).to_string(index=False) if not c.startswith("1") else S[C].head(12).round(3).to_string(index=False) + f"\n   ... ({len(S)} total; showing 12 most trusted)")
T.to_csv(os.path.join(env.LOCAL, "ar2_pairs.csv"), index=False)
