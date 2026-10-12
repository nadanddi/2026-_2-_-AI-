# -*- coding: utf-8 -*-
"""AR1 (2026-10-13 집 클로드; 사용자: '확인 가능/불가인 애들을 모델이 구분해야겠네').  서술(신호 확인) — 판정 아님.
질문: 앵커 짝 하나하나가 'EC 를 망치는 짝'인지 평가 때 볼 수 있는 정보만으로 가려낼 수 있나.
후보 짝 모음(거울 CV 기하): v7 짝(p ≥ .6·정답 기록) ∪ v12 짝(p ≥ .6·정답 기록·날짜 불일치 아님), (rec, prev) 중복 제거.
정답(학습이라 앎): bad = |짝 23시 EC − 대상 실제 0시 EC| > .10  (진짜 같은 동 연속이면 자정 EC 차가 거의 0; ECJ1b 손해 짝은 .3~1.1).
특징(평가 때 알 수 있는 것만): p, v7 이 고른 짝인지, v12 가 고른 짝인지, 두 링커 일치, HP2 일치, 다른 ID 짝인지, 날짜 상태(ok/unknown),
  자정 실내 3변수 점프(대상 0시 − 짝 23시: 온도·습도·CO2, 절댓값), 0시 운영 지문 거리, |짝 23시 EC − 대상 0시 예측 sg0|(DIAG10 CUR 시드 평균).
보고: 특징별 AUC, 로지스틱(표준화) 거울 블록 단위 교차적합 AUC — 다른 ID 특징 있음/없음 둘 다, 그리고 '맞은 짝을 95% 남길 때 나쁜 짝을 몇 개 거르나'.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])
from feat4 import EC  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
blk = lambda r: f"{r[:3]}_{(int(r[4:7]) - NUM[FARM == r[:3]].min()) // 10}"
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
v7p = {r: (pv, p) for r, pv, p, l in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled) if l and p >= .6 and isinstance(pv, str)}
v12p = {r: (pv, p, ds) for r, pv, p, ds, l in zip(PM.rec, PM.prev, PM.p, PM.date_status, PM.labelled) if l and p >= .6 and ds != "mismatch"}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]
SG0 = O.assign(rec=O.row_id.str[:7]).groupby("rec").sg.mean().to_dict()
rows = []
for r in sorted(set(v7p) | set(v12p)):
    for pv in {x[0] for x in (v7p.get(r), v12p.get(r)) if x}:
        b, a = idx[r], idx[pv]
        if not np.isfinite(EC[b, 0]) or not np.isfinite(EC[a, 23]) or r not in SG0: continue
        h = hp2(b, mirror_lab(r))
        rows.append(dict(rec=r, prev=pv, blk=blk(r), pass2=int(r[4:7]) >= 179,
                         p=max(v7p[r][1] if r in v7p and v7p[r][0] == pv else 0, v12p[r][1] if r in v12p and v12p[r][0] == pv else 0),
                         by_v7=float(r in v7p and v7p[r][0] == pv), by_v12=float(r in v12p and v12p[r][0] == pv),
                         hp2_agree=float(h == pv), date_ok=float(r in v12p and v12p[r][0] == pv and v12p[r][2] == "ok"), cross_id=float(r[:3] != pv[:3]),
                         jT=abs(IN_[b, 0, 0] - IN_[a, 23, 0]), jH=abs(IN_[b, 0, 1] - IN_[a, 23, 1]), jC=abs(IN_[b, 0, 2] - IN_[a, 23, 2]),
                         fp=float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4))), gapA=abs(EC[a, 23] - SG0[r]),
                         jump_true=abs(EC[a, 23] - EC[b, 0])))
T = pd.DataFrame(rows); T = T.fillna(T.median(numeric_only=True)); T["both"] = T.by_v7 * T.by_v12; T["bad"] = (T.jump_true > .10).astype(int)
print(f"candidate pairs {len(T)} | bad {T.bad.sum()} (pass-2 {T[T.pass2].bad.sum()} of {T.pass2.sum()}) | jump_true quantiles good:",
      T[T.bad == 0].jump_true.quantile([.5, .9, .99]).round(3).tolist())
FEAT = ["p", "by_v7", "by_v12", "both", "hp2_agree", "date_ok", "cross_id", "jT", "jH", "jC", "fp", "gapA"]
print("\nsingle-feature AUC (bad = 1; >.5 means larger value -> more bad):")
print({f: round(roc_auc_score(T.bad, T[f]), 3) for f in FEAT})
pd.set_option("display.width", 220)
print("\nbad pairs:"); print(T[T.bad == 1][["rec", "prev", "pass2", "p", "by_v7", "by_v12", "hp2_agree", "date_ok", "cross_id", "jT", "jH", "jC", "fp", "gapA", "jump_true"]].round(3).to_string(index=False))
for nm, F in (("with cross_id", FEAT), ("without cross_id", [f for f in FEAT if f != "cross_id"])):
    oof = np.zeros(len(T))
    for g in T.blk.unique():
        tr, te = T.blk != g, T.blk == g
        if T.bad[tr].nunique() < 2: continue
        Z = T[F].values.astype(float); mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        m = LogisticRegression(C=.5, max_iter=2000).fit((Z[tr] - mu) / sd, T.bad[tr]); oof[te.values] = m.predict_proba((Z[te] - mu) / sd)[:, 1]
    good = T.bad.values == 0; thr = np.quantile(oof[good], .95)
    print(f"\nlogistic ({nm}) block-CV AUC {roc_auc_score(T.bad, oof):.3f} | keep 95% of good -> bad removed {int(((oof > thr) & ~good).sum())}/{int((~good).sum())}"
          f" | pass-2 AUC {roc_auc_score(T.bad[T.pass2], oof[T.pass2.values]) if T[T.pass2].bad.nunique() > 1 else float('nan'):.3f}")
    T["oof_" + nm.split()[0]] = oof
print("\nbad pairs oof score rank (1 = most suspicious):"); T["rk"] = T["oof_with"].rank(ascending=False)
print(T[T.bad == 1][["rec", "prev", "oof_with", "oof_without", "rk"]].round(3).to_string(index=False))
T.to_csv(os.path.join(env.LOCAL, "ar1_pairs.csv"), index=False)
