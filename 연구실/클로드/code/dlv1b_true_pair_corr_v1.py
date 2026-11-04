# -*- coding: utf-8 -*-
"""DLV1b (2026-11-04 연구실 클로드; 사용자 '왜 기준이 다른 49개 온실 중앙값인가, 써야 하는 건 F13·F47').  서술(판정 없음).  온도.
DLV1(집/클로드/research/dlv1_day_level_offsets_all_farms_v1.py)과 같은 기준 모델(SELF_LR, 기록 번호 5덩어리 OOF)로
F13·F47 의 기록별 하루 평균 오차 m 을 구하고, '어제와 이어짐' 상관을 두 가지 짝으로 잰다.
  (가) 번호 이웃: 같은 ID 에서 기록 번호가 1 차이(DLV1 과 같음, F13 .15·F47 .29 재현 확인용)
  (나) 진짜 전날: linker_v3/truth.pkl 의 확실한 짝(cert, 190쌍, ID 가 바뀌는 짝 포함)
Run:  PYTHONPATH="" python -u 연구실/클로드/code/dlv1b_true_pair_corr_v1.py
"""
import os, sys
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "집", "클로드", "research"))
sys.path.insert(0, RES); os.chdir(RES)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(RES, "dlv1_day_level_offsets_all_farms_v1.py"), encoding="utf-8").read()
src = src[:src.index("def stats(T):")]
__file__ = os.path.join(RES, "dlv1b.py")
exec(compile(src, "dlv1pre", "exec"))

def day_means(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]; R["e"] = R.pred - R.sub_temp
    return R.groupby("day").e.mean()

M = {T: day_means(T) for T in ("F13", "F47")}
def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())
for T in ("F13", "F47"):
    m = M[T].sort_index(); cons = m.index.to_series().diff() == 1
    print(f"(가) 번호 이웃 {T}: r = {corr(m.values[1:][cons.values[1:]], m.values[:-1][cons.values[1:]])}")
tp = pd.read_pickle(os.path.join(RES, "local", "linker_v3", "truth.pkl")); tp = tp[tp.cert]
get = lambda f, n: M[f].get(n, np.nan)
y = [get(f, n) for f, n in zip(tp.fb, tp.nb)]; t = [get(f, n) for f, n in zip(tp.fa, tp.na)]
print(f"(나) 진짜 전날 전체: r, n = {corr(y, t)}")
for lab, msk in (("같은 ID 짝", tp.fb.values == tp.fa.values), ("ID 가 바뀌는 짝", tp.fb.values != tp.fa.values)):
    print(f"(나) {lab}: r, n = {corr(np.array(y)[msk], np.array(t)[msk])}")
for T in ("F13", "F47"):
    msk = (tp.fa.values == T)
    print(f"(나) 오늘이 {T} 인 짝: r, n = {corr(np.array(y)[msk], np.array(t)[msk])}")
