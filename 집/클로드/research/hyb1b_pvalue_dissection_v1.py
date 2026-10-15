# -*- coding: utf-8 -*-
"""HYB1b (2026-10-16 집 클로드; 사용자 'HYB1 확인해봐. p 값 왜 높지?').  서술(판정 바꾸지 않음), 실행 전 커밋.  온도.
HYB1(6.992) 을 같은 코드로 다시 계산해 HYB vs M1 차이를 분해: (1) 셀 간 상관(16칸이 독립 증거인가) (2) 바뀐 기록 수·기록별 dSSE 분포(도움/해, 상위 몫)
(3) 블록(농장 x 5일) 단위 dSSE 분포·0 블록 비율·부트스트랩 평균/표준오차 (4) 바뀐 기록의 연결 확신·dT 구간별 기록 dSSE.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hyb1b_pvalue_dissection_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "hyb1_chain_m1_gate_hybrid_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "hyb1_chain_m1_gate_hybrid_v1.py")', '__file__ = os.path.join(HERE, "hyb1b_pvalue_dissection_v1.py")')
src = src.replace("    RES[L] = dict(P=P, y=y, t=M.in_temp.values, bk=", "    RES[L] = dict(P=P, rec=key, PLw=PLw, PLn=PLn, y=y, t=M.in_temp.values, bk=")
exec(compile(src[:src.index("def cellch(L, a, b):")], "hyb1", "exec"))
pd.set_option("display.width", 220)
print("\n(1) correlation between cells of per-row dSSE (HYB - M1), per layout")
for L in ("CV8", "N", "CK2"):
    R_ = RES[L]; y = R_["y"]; D = np.array([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(R_["P"]["HYB"], R_["P"]["M1"])])
    c = np.corrcoef(D); print(f"  {L}: mean off-diagonal corr {c[np.triu_indices(4, 1)].mean():.3f}")
rows = []
for L in ("CV8", "N", "CK2", "EL1"):
    R_ = RES[L]; y = R_["y"]; d = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(R_["P"]["HYB"], R_["P"]["M1"])], 0)
    S = pd.Series(d).groupby(R_["rec"]).sum(); ch = S[S.abs() > 1e-9]
    extra = [r for r, v in R_["PLw"].items() if v[0] == "CH" and r not in R_["PLn"]]
    for r in ch.index:
        p, q, dt = R_["PLw"][r][1:] if r in R_["PLw"] else (None, np.nan, np.nan)
        rows.append(dict(layout=L, rec=r, dsse=ch[r], p=q, dt=dt, extra=r in extra))
    print(f"(2) {L}: records changed {len(ch)} (extra chains {len(extra)}) | helped {int((ch < 0).sum())} harmed {int((ch > 0).sum())} | sum {ch.sum():+.2f} | top3 |share| {ch.abs().nlargest(3).sum() / max(ch.abs().sum(), 1e-9):.2f} | best {ch.min():+.2f} worst {ch.max():+.2f}")
T = pd.DataFrame(rows)
print("   largest record effects:"); print(T.reindex(T.dsse.abs().sort_values(ascending=False).index).head(12).round(3).to_string(index=False))
print("\n(3) block-level (farm x 5-day, averaged over layouts) dSSE")
per = {L: pd.Series(np.mean([(a - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for a, b in zip(RES[L]["P"]["HYB"], RES[L]["P"]["M1"])], 0)).groupby(RES[L]["bk"]).mean() for L in ("CV8", "N", "CK2")}
U = pd.concat(per, axis=1).mean(axis=1); nz = U[U.abs() > 1e-12]
print(f"  blocks {len(U)} | blocks with any change {len(nz)} ({len(nz) / len(U):.0%}) | helped {int((nz < 0).sum())} harmed {int((nz > 0).sum())} | mean {U.mean():+.5f} se {U.std(ddof=1) / np.sqrt(len(U)):.5f} | t {U.mean() / (U.std(ddof=1) / np.sqrt(len(U))):+.2f}")
print("  changed blocks sorted:", nz.sort_values().round(4).to_dict())
print("\n(4) record dSSE by link confidence / dT (changed records)")
T["pbin"] = pd.cut(T.p, [0, .5, .65, .8, 1.01]); T["dtbin"] = pd.cut(T.dt, [-.01, .5, 1.5, 3.01])
print(T.groupby("pbin", observed=True).dsse.agg(["size", "sum", "mean"]).round(3).to_string()); print(T.groupby("dtbin", observed=True).dsse.agg(["size", "sum", "mean"]).round(3).to_string())
