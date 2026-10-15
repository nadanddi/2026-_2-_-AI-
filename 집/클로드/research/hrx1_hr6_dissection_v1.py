# -*- coding: utf-8 -*-
"""HRX1 (2026-10-15 집 클로드; 사용자 '이전 모델을 뜯어봐서 살릴 부분만, 5점 규정 확인').  서술 분해(판정 없음), 실행 전 커밋.  EC.
HR6(6.919) = HR5 + 어제 EC 특징(AF = 짝 23시 EC − 기본 누적평균, AP = p0, AH = 썼는가), 짝 = v15R 거울(X: 미러 기하·번호 특징) p0 ≥ .3.
저장 행 hr6_rows_v1.csv.gz(검증기 x 시드 x 행: base, hr5, hr6) 를 기록 단위로 나눠 HR6 − HR5 dSSE(2차 행, 시드 평균) 를 분해:
  링크 대역: 없음(p0 < .3 또는 정답 없는 짝) / .3~.8 / ≥ .8;  짝 맞음(cert 정답 짝과 같음)/틀림/정답 짝 미상;  FL 링커(fl3 학습표)와 같은 짝인가;  고EC/보통.
  (A 사용 여부는 폴드 검증 집합을 저장 안 해서 '짝이 정답 있는 학습 기록·잠금 아님' 으로 근사)
목적: 이득이 '확신 있고 맞는 어제' 에서 나오는지 → 5점판(FL 확신 짝 p ≥ .8 만) 재구성의 근거.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hrx1_hr6_dissection_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import TL7, rid, TRAIN, FARM, NUM, N, EC  # noqa: E402
import numpy as np, pandas as pd
IDX = {rid(i): i for i in range(N)}; CERT = {rid(int(b)): rid(int(a)) for a, b in zip(TL7.a, TL7.b)}
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); V15 = {r: (p, float(q)) for r, p, q in zip(PM.rec, PM.prev0, PM.p0) if isinstance(p, str)}
FL = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FL.rec, FL.prev, FL.p)}
LOCK = set(pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"), usecols=["rec", "lock"]).query("lock").rec)
R = pd.read_csv(os.path.join(HERE, "hr6_rows_v1.csv.gz")); R = R[R.p2]
R["dd"] = (R.hr6 - R.y) ** 2 - (R.hr5 - R.y) ** 2
G = R.groupby(["validator", "rec"]).agg(dd=("dd", "sum"), hi=("hi", "first"), n=("dd", "size"), nseed=("seed", "nunique")).reset_index(); G["dd"] /= G.nseed
def band(r):
    p, q = V15.get(r, (None, 0.0))
    if p is None or q < .3 or p not in IDX or not TRAIN[IDX[p]] or p in LOCK or not np.isfinite(EC[IDX[p], 23]): return "0 none"
    return "2 >=.8" if q >= .8 else "1 .3-.8"
def correct(r):
    p = V15.get(r, (None, 0))[0]
    if r not in CERT: return "unknown"
    return "right" if p == CERT[r] else "wrong"
G["band"] = G.rec.map(band); G["link"] = G.rec.map(correct); G["fl_same"] = [FLK.get(r, (None, 0))[0] == V15.get(r, (None, 0))[0] for r in G.rec]
G["fl_conf"] = [FLK.get(r, (None, 0))[1] >= .8 and FLK.get(r, (None, 0))[0] == V15.get(r, (None, 0))[0] for r in G.rec]
pd.set_option("display.width", 200)
for v in ("P2LOO", "EL1", "DIAG10"):
    g = G[G.validator == v]; print(f"\n== {v} (pass-2 records {len(g)}, total dSSE {g.dd.sum():+.2f}; negative = HR6 better)")
    print(g.groupby(["band", "link"]).agg(records=("rec", "size"), dSSE=("dd", "sum"), high=("hi", "sum")).round(2).to_string())
    print("  by EC level:", g.groupby("hi").dd.sum().round(2).to_dict(), "| FL-confident same pair:", g.groupby("fl_conf").dd.sum().round(2).to_dict(), "(records", g.groupby("fl_conf").size().to_dict(), ")")
    print("  top gains:", g.nsmallest(4, "dd")[["rec", "dd", "band", "link", "hi"]].round(2).values.tolist())
    print("  top losses:", g.nlargest(4, "dd")[["rec", "dd", "band", "link", "hi"]].round(2).values.tolist())
