# -*- coding: utf-8 -*-
"""HD1b (2026-10-14 집 클로드; 사용자: '가려내는 능력 보려면 비율 어떻게' → '그렇게 해봐').  서술, 실행 전 커밋.  EC.
HD1 의 OOF 판별 확률(local/hd1_probs_v1.pkl, 학습 기록 전부)로:
  1단계 능력(비율 무관): AUC, 헛경보율(FPR) 1%·3%·5%·10% 로 문턱을 정했을 때의 포괄률(recall).  문턱 = 보통 날 확률의 (1 − FPR) 분위수.
  2단계 쓸모: 같은 문턱으로, 고EC 비율 9%(학습 전체 35/400 근처)·11%(2차 학습 5/46) 일 때 기대 정확도(precision)
             = recall·π / (recall·π + FPR_실제·(1 − π)),  그리고 보통 날 100일당 헛경보 수.  비율은 학습 자료에서만(리더보드 추정 안 씀).
  판별기 R·ET, 시점 h 9·23, 배치 B20·I10 (HD1 과 같음).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hd1b_detection_recall_at_fpr_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
REC = [rid(i) for i in range(N) if TRAIN[i] and np.isfinite(DM[i])]; Yh = np.array([DM[idx[r]] >= 1.0 for r in REC])
P = pd.read_pickle(os.path.join(env.LOCAL, "hd1_probs_v1.pkl")); rows = []
for sc in ("B20", "I10"):
    for mdl in ("R", "ET"):
        for h in (9, 23):
            p = np.array([P[(sc, mdl, h, i)] for i in range(len(REC))]); neg = p[~Yh]; pos = p[Yh]
            for fpr in (.01, .03, .05, .10):
                th = np.quantile(neg, 1 - fpr); fa = float((neg > th).mean()); rec = float((pos > th).mean())
                d = dict(scheme=sc, model=mdl, hour=h, AUC=roc_auc_score(Yh, p), FPR_target=fpr, FPR_actual=fa, recall=rec, caught=int((pos > th).sum()), of=len(pos))
                for pi in (.09, .11): d[f"precision@{int(pi * 100)}%"] = rec * pi / (rec * pi + fa * (1 - pi)) if rec + fa > 0 else np.nan
                d["false_alarms_per_100_normal"] = 100 * fa; rows.append(d)
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
