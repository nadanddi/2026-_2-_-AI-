# -*- coding: utf-8 -*-
"""DLV1c (2026-11-04 연구실 클로드; 사용자 '잘 맞춘 날, 통째로 어긋난 날, 완전히 틀린 날… 비율은?').  서술(판정 없음), 설명용 날 분류.  온도.
DLV1 과 같은 기준 모델(SELF_LR, 기록 번호 5덩어리 OOF)로 F13·F47 의 기록(하루)마다
  하루 RMSE, 그날 오차 중 '통째로 밀린 몫' = m^2 / mean(e^2)  (m = 그날 평균 오차)
을 구하고, 설명용 문턱(사전에 정함, 근거 없음 — 읽기 쉬운 반올림 값)으로 날을 나눈다.
  잘 맞은 날: 하루 RMSE < 0.5℃
  나머지 중 통째로 밀린 날: 밀린 몫 ≥ 0.6 / 시간별로 흔들린 날: 밀린 몫 < 0.3 / 섞인 날: 그 사이
  크게 틀린 날(겹쳐 표시): 하루 RMSE ≥ 1.5℃
날 수 비율과 전체 제곱 오차 중 몫을 함께 낸다. 문턱을 바꾼 결과도 한 줄씩.
Run:  PYTHONPATH="" python -u 연구실/클로드/code/dlv1c_day_types_v1.py
"""
import os, sys
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "집", "클로드", "research"))
sys.path.insert(0, RES); os.chdir(RES)
import env  # noqa: F401
import numpy as np, pandas as pd
src = open(os.path.join(RES, "dlv1_day_level_offsets_all_farms_v1.py"), encoding="utf-8").read()
src = src[:src.index("def stats(T):")]
__file__ = os.path.join(RES, "dlv1c.py")
exec(compile(src, "dlv1pre", "exec"))

def oof(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]; R["e"] = R.pred - R.sub_temp
    return R

def classify(R, good, hi, lo):
    g = R.groupby("day").e; m = g.mean(); ms = g.apply(lambda s: float(np.mean(s ** 2))); n = g.size()
    rmse = np.sqrt(ms); sh = m ** 2 / ms; sse = ms * n
    typ = np.where(rmse < good, "잘 맞음", np.where(sh >= hi, "통째로 밀림", np.where(sh < lo, "시간별 흔들림", "섞임")))
    T = pd.DataFrame(dict(rmse=rmse, sh=sh, sse=sse, typ=typ))
    out = T.groupby("typ").agg(days=("sse", "size"), sse=("sse", "sum")); out["날 비율%"] = 100 * out.days / len(T); out["오차 몫%"] = 100 * out.sse / T.sse.sum()
    big = rmse >= 1.5
    return out[["days", "날 비율%", "오차 몫%"]].round(1), int(big.sum()), round(100 * big.mean(), 1), round(100 * float(sse[big].sum() / sse.sum()), 1), float(np.median(rmse))

for F in ("F13", "F47"):
    R = oof(F)
    tab, nb, pb, eb, med = classify(R, 0.5, 0.6, 0.3)
    print(f"\n== {F}: 하루 {R.day.nunique()}개, 하루 RMSE 중앙 {med:.2f}℃, 전체 RMSE {rm(R.e):.3f}℃")
    print(tab.to_string())
    print(f"크게 틀린 날(하루 RMSE ≥ 1.5℃): {nb}일 ({pb}%), 전체 제곱 오차 중 {eb}%")
    for good, hi, lo in ((0.4, 0.6, 0.3), (0.6, 0.6, 0.3), (0.5, 0.7, 0.2), (0.5, 0.5, 0.4)):
        t2 = classify(R, good, hi, lo)[0]
        print(f"  문턱 바꿈(잘 맞음<{good}, 밀림≥{hi}, 흔들림<{lo}): " + ", ".join(f"{k} {v:.0f}%일/{w:.0f}%오차" for k, v, w in zip(t2.index, t2["날 비율%"], t2["오차 몫%"])))
