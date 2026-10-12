# -*- coding: utf-8 -*-
"""ST1 (2026-10-12 집 클로드; 사용자: '날짜는 외기가 이어지는 것만, 동은 실내가 이어지는 것만 보면 되지 않나').  서술(학습 없는 2단계 규칙).
후보 = v10 후보 집합(features10, 거울 CV 기하, 0시 판단).
  1단계 날짜: 후보마다 외기 자정 연속 비용 oj(전날 23시 외기 외삽 vs 오늘 0시 외기, 4변수 표준화, v7 공식); 후보들을 달력 날짜(DM11)로 묶어 날짜별 최소 oj → 가장 작은 날짜 선택.
  2단계 동: 그 날짜 후보 중 실내·구동기 자정 연속 비용 Σ_j min(dv_j, 6) 최소.
  확신 = 2단계 1·2등 비용 차(margin), 1단계 1·2등 날짜 차(date_margin).
보고: 확실 연결(cert) 기준 이을 수 있는 기록에서 1순위 정확도, 날짜 맞음 비율, 날짜 맞을 때 동 맞음 비율; margin 문턱별 정확도·포괄률; v10·META 와 비교.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {int(b): int(a) for a, b in zip(TLa.a1, TLa.b)}; CERT = {int(b): bool(c) for b, c in zip(TLa.b, TLa.cert)}
rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b] or b not in TP or not CERT[b] or not lab[TP[b]]: continue
            D, _ = features10(b, 0, lab, removed=rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); oj = D.oj.values; dv = np.minimum(D[[f"v{j}" for j in range(8)]].values, 6).sum(1)
            dd = pd.DataFrame(dict(a=a, oj=oj, dv=dv, day=DAY[a])).dropna(subset=["day"])
            byd = dd.groupby("day").oj.min().sort_values()
            dsel = byd.index[0]; dmar = (byd.iloc[1] - byd.iloc[0]) if len(byd) > 1 else 9
            S2 = dd[dd.day == dsel].sort_values("dv"); pick = int(S2.a.iloc[0]); mar = (S2.dv.iloc[1] - S2.dv.iloc[0]) if len(S2) > 1 else 9
            t = TP[b]
            rows.append(dict(rec=rid(b), p2=NUM[b] >= 179, date_ok=dsel == DAY[t], ok=pick == t, lab=bool(lab[pick]), margin=mar, date_margin=dmar, n_day=len(S2)))
T = pd.DataFrame(rows)
print(f"cert linkable records: {len(T)} | top-1 correct {T.ok.mean():.3f} | date correct {T.date_ok.mean():.3f} | house correct when date correct {T[T.date_ok].ok.mean():.3f} | pass-2 top-1 {T[T.p2].ok.mean():.3f} (n {int(T.p2.sum())})")
for th in (0, .5, 1, 2, 3):
    S = T[(T.margin >= th) & T.lab]; print(f"   house margin >= {th}: anchors {len(S)}, precision {S.ok.mean() if len(S) else float('nan'):.3f}, recall {S.ok.sum()/len(T):.2f}")
print("reference (cert, same linkable set definition, from LR2): v10 >=.8 38 anchors prec 1.000 rec .58 | META >=.7 44 prec 1.000 rec .68")
