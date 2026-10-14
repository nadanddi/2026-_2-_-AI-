# -*- coding: utf-8 -*-
"""HR5E (2026-10-14 집 클로드).  서술(정답 없음).  HR5(6.890) 보정기를 학습 정답 360 전부로 학습해 평가 60 에 적용했을 때의 이동량.
기본 = 14회차 EC(submission_23 sub_ec) — 평가 행 특징 = 자기 기록 0..h' 입력 + 14회차 예측 요약(HR5 와 같은 feat).  λ = .5(HR5 선택).
보고: 기록별 하루 평균 이동, 9시 이후 평균 이동, 행 RMS, 14회차 >= 1 기록 vs 보통 기록.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "hr5e_eval_exposure_v1.py"); LAM = .5
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14["h"] = P14.row_id.str[8:].astype(int)
P14 = P14[P14.rec.str[:3].isin(["F13", "F47"])].sort_values(["rec", "h"])
models = {}
for h in HS:
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True); m.fit(TF[h], TD); models[h] = m
rows = []
for r, g in P14.groupby("rec"):
    base = g.sub_ec.values; Q = np.stack([feat(INP[r], base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS]))
    dh = {h: float(models[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}; sh = LAM * np.array([dh[max(x for x in HS if x <= hh)] for hh in range(24)])
    rows.append(dict(rec=r, cur=base.mean(), shift_day=sh.mean(), shift_after9=sh[9:].mean(), shift_23h=sh[23]))
E = pd.DataFrame(rows); pd.set_option("display.width", 200); print(E.round(3).sort_values("shift_day").to_string(index=False))
allsh = E.shift_day.values
print(f"\nrecords {len(E)} | mean shift {allsh.mean():+.3f} | cur >= 1: mean shift {E[E.cur >= 1].shift_day.mean():+.3f} (n {int((E.cur >= 1).sum())}) | others {E[E.cur < 1].shift_day.mean():+.3f} | |day shift| > .05: {int((E.shift_day.abs() > .05).sum())}")
E.to_csv(os.path.join(HERE, "hr5e_eval_exposure_v1.csv"), index=False)
