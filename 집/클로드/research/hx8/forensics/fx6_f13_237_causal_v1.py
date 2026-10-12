# -*- coding: utf-8 -*-
# hx8/forensics fx6 v1: F13_237 시각 인과판 '소거 + 양쪽 괄호' 계산(서술, 평가 1기록). 제출물 아님.
# h시 행은 F13_237 자기 0..h시 입력만 + 학습 기록(전 시각·정답·뒤 날짜 포함) 사용. 다른 ID 평가 기록·같은 ID 뒷번호 입력 미사용.
# 절차(사전 고정): (1) 날짜 쌍둥이 = 외기 0..h 가 F13_237 과 같은 학습 기록. (2) 쌍둥이(학습, 하루 전체 사용)는 그날의 다른 물리 동 →
#   그 동들은 F13_237 후보에서 소거. (3) 다음날 학습 묶음 = 쌍둥이 23시 외기와 0시 외기가 가장 가까운 학습 묶음; 이틀 전 묶음 =
#   전날 묶음(쌍둥이 0시 ↔ 23시)의 0시 ↔ 23시 연속. (4) 앞·뒤 묶음에서 쌍둥이와 지문이 가장 가까운 기록(= 소거된 동의 앞뒤)을 빼고
#   남은 기록 중 F13_237 의 0..h 구동기 지문 최근접을 앵커로. (5) 값 = 앞 23시 ~ 뒤 0시 직선 보간.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6; sdO = np.nanstd(X[tr].reshape(-1, 4), 0)
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
B = [v for v in key.values()]
def acd(x, y, h=23): return np.nanmean(np.abs(AC[x, :h + 1] - AC[y, :h + 1]) / sdA)
def next_bundle(v): return min((b for b in B if b != v), key=lambda b: np.nanmean(np.abs(X[b[0], 0] - X[v[0], 23]) / sdO))
def prev_bundle(v): return min((b for b in B if b != v), key=lambda b: np.nanmean(np.abs(X[b[0], 23] - X[v[0], 0]) / sdO))
c = ix["F13_237"]
sub = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
p13 = sub[sub.row_id.str.startswith("F13_237")].sort_values("row_id").sub_ec.values
rows = []
for h in range(24):
    tw = [i for i in tr if (np.abs(X[i, :h + 1] - X[c, :h + 1]) < 1e-6).all()]
    if not tw: rows.append(dict(h=h, twins="")); continue
    v = next(b for b in B if tw[0] in b)
    nb = next_bundle(v); pb1 = prev_bundle(v); pb2 = prev_bundle(pb1)
    # remove successors/predecessors of the occupied dongs (twins) by full-day fingerprint (training only)
    def rest(bund, ref_list):
        occ = {min(bund, key=lambda k: acd(t, k)) for t in ref_list} if len(bund) > len(ref_list) else set()
        return [k for k in bund if k not in occ]
    nr = rest(nb, tw); pr = rest(pb2, tw)
    a = min(pr, key=lambda k: acd(c, k, h)) if pr else None
    d = min(nr, key=lambda k: acd(c, k, h)) if nr else None
    val = np.nan
    if a is not None and d is not None:
        val = EC[a, 23] + (EC[d, 0] - EC[a, 23]) * (48 + h - 23) / (72 - 23)
    rows.append(dict(h=h, twins=" ".join(rid[tw]), prev2=" ".join(rid[pb2]), next=" ".join(rid[nb]),
                     prev_rest=" ".join(rid[pr]), next_rest=" ".join(rid[nr]),
                     a=rid[a] if a is not None else "", d=rid[d] if d is not None else "", fb=val, p13=p13[h]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "fx6_f13_237_causal_v1.csv"), index=False)
pd.set_option("display.width", 250)
print(R[["h", "twins", "prev2", "next", "prev_rest", "next_rest", "a", "d", "fb", "p13"]].round(3).to_string(index=False))
print(f"\nday mean fb {np.nanmean(R.fb):.3f} vs 13th {p13.mean():.3f}; RMS change {np.sqrt(np.nanmean((R.fb - R.p13) ** 2)):.3f}")
# what the other remaining dong would give (robustness: both remaining dongs normal?)
print("bracket values of all remaining candidates (prev 23h / next 0h):",
      {rid[k]: round(float(EC[k, 23]), 3) for k in pr}, {rid[k]: round(float(EC[k, 0]), 3) for k in nr})
