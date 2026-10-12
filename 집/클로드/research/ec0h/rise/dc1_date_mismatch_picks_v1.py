# -*- coding: utf-8 -*-
"""DC1 (2026-10-12 집 클로드, 서술): 짝찾기 v7 의 전날 선택이 달력(DM11)상 '대상 날짜 − 1'이 아닌 경우를 셈 — 학습 거울 CV(cv_all400_choice_v7m) 와 평가 60.
학습 쪽은 진실(truth a1) 대비 정확도를 날짜 일치/불일치로 나눠 보고, 날짜 불일치 선택의 0시 EC 차(선택 23시 − 정답 0시)도 표시."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); truth = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}; cert = {rid(int(b)): bool(c) for b, c in zip(TL.b, TL.cert)}
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"))
L["d_t"] = L.rec.map(DAY); L["d_p"] = L.prev.map(DAY); L["date_ok"] = (L.d_p == L.d_t - 1)
L["truth"] = L.rec.map(truth); L["cert"] = L.rec.map(cert); L["ok"] = L.prev == L.truth
print("TRAIN mirror-CV picks:", len(L), "| date known both:", int(L[["d_t", "d_p"]].notna().all(1).sum()))
K = L[L[["d_t", "d_p"]].notna().all(1)]
print("date-correct picks: %.2f | accuracy vs truth (cert) when date ok %.2f (n %d) vs date wrong %.2f (n %d)" % (
    K.date_ok.mean(), K[K.date_ok & K.cert].ok.mean(), int((K.date_ok & K.cert).sum()), K[~K.date_ok & K.cert].ok.mean(), int((~K.date_ok & K.cert).sum())))
print("date-wrong picks with p >= .8:", int((~K.date_ok & (K.p >= .8)).sum()), "| among them cert-truth correct:", int((~K.date_ok & (K.p >= .8) & K.cert & K.ok).sum()))
print("day difference distribution of date-wrong picks:", (K[~K.date_ok].d_t - K[~K.date_ok].d_p).value_counts().head(10).to_dict())
E = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"))
E["d_t"] = E.row.map(DAY); E["d_p"] = E.prev_h0.map(DAY); E["date_ok"] = E.d_p == E.d_t - 1
print("\nEVAL 60: date known both %d | date-correct picks %d | date-wrong %d" % (int(E[["d_t", "d_p"]].notna().all(1).sum()), int(E.date_ok.sum()), int((~E.date_ok & E[["d_t", "d_p"]].notna().all(1)).sum())))
print(E[~E.date_ok][["row", "prev_h0", "p_h0", "prev_is_labelled", "xf", "anchor_ec23", "d_t", "d_p"]].to_string(index=False))
C = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_confident_1step_anchor_v7.csv"))
C["d_t"] = C.row.map(DAY); C["d_p"] = C.prev_h0.map(DAY)
print("\ncurrently ANCHORED eval records with date-wrong predecessor:", C[C.d_p != C.d_t - 1][["row", "prev_h0", "p_h0", "anchor_ec23", "d_t", "d_p"]].to_string(index=False))
