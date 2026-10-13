# -*- coding: utf-8 -*-
"""SV3c (2026-10-14 집 클로드; 사용자: 'RMSE 검사까지 가능한가?').  서술 — SV3 구조 일치 검증기(2차 val 21기록, 배치 9)에서 후보별 절대 RMSE 와 95% 구간(배치 재표집 5,000), 시드 평균 예측.
후보 정의는 SV3b 와 같음(기본 = 실제 파이프라인 SG2 후, EA2 = v7 짝 앵커, EA2B5 = 지금 제출 구성에 가까움, LKH, 정답 앵커 = 상한)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "sv3b_structure_validator_judge_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "sv3b_structure_validator_judge_v1.py")
exec(src[:src.index("RES = {}")])
__file__ = os.path.join(HERE, "sv3c_rmse_table_v1.py")
P = {"기본 파이프라인(SG2 후)": {s: G[f"fin_{s}"].values for s in SEEDS}, "+ EA2 앵커": apply("EA2"), "+ EA2 + B5 (제출 구성 근사)": apply("EA2B5"),
     "+ 시각별 링커 앵커(LKH)": apply("LKH"), "정답 앵커 100% (상한, 못 씀)": apply("DOSE", 1.0)}
dr = G.draw.values; ks = np.unique(dr); hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec]); out = []
for nm, Pd in P.items():
    p = np.mean([Pd[s] for s in SEEDS], axis=0)
    for sub, m in (("전체", np.ones(len(G), bool)), ("보통 날", ~hi), ("고EC 날", hi)):
        e2 = pd.Series(((p - y) ** 2)[m]); n = pd.Series(np.ones(m.sum())); a = e2.groupby(dr[m]).sum(); b = n.groupby(dr[m]).sum(); kk = a.index.values
        boots = [np.sqrt(a[s].sum() / b[s].sum()) for s in [rng.choice(kk, len(kk)) for _ in range(5000)]]
        out.append(dict(cand=nm, subset=sub, records=int(pd.Series(G.rec[m]).nunique()), rmse=float(np.sqrt(a.sum() / b.sum())), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5))))
T = pd.DataFrame(out); pd.set_option("display.width", 200); print(T.round(4).to_string(index=False)); T.to_csv(os.path.join(HERE, "sv3c_rmse_table_v1.csv"), index=False)
