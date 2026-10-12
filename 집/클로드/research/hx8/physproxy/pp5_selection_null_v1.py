# -*- coding: utf-8 -*-
"""PP5 (2026-10-13 집 클로드, hx8/physproxy).  PP4 (B-L) 에서 M_AHxc 가 DIAG10 3시드·BUF/LOSO+ 모두 RMSE 를 낮췄지만
AHx 는 PP1 에서 같은 40기록으로 21종 중 '가장 좋은 것'으로 고른 특징 → PP4 의 단일 특징 확인은 독립 검증이 아님(선택 편향).
사전 고정 점검(서술):
 (1) 선택 보정 기준선: 같은 PP4 L 가짜 평가(LOSO+, BUF; seed 47)를 21종 파생량 각각에 돌려 '최대 이득'을 통계로,
     특징 표를 기록 순열 / seg 블록 순열(seg 사이 순환 배치) 각 300회 → 95% 분위.  AHx 실제 이득이 둘 다 넘어야 '선택 보정 후 통과'.
 (2) 6.697 류 단순 요약(하루 평균 in_temp, out_temp, in_hum, out_hum, act_vent, act_circfan)도 같은 틀로 — AHx 가 단순 요약 재포장인지.
 (3) 영향 기록: M_AHxc 이득이 어느 기록에서 나오나(오차 제곱 감소 상위 5), |y−q| 상위 3기록 빼고 다시.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp4_pseudoeval_fix_v1 import *   # noqa
import warnings; warnings.filterwarnings("ignore")

SIMPLE = pd.DataFrame({r: {v: np.nanmean(A_tr[r][v]) for v in ["in_temp", "out_temp", "in_hum", "out_hum", "act_vent", "act_circfan"]} for r in A_tr}).T

Q47 = qday_of("DIAG10", 47)
_FOLDS = {}
def folds(mode):
    if mode not in _FOLDS:
        out = []
        for i, t in enumerate(TGT):
            ex = excl(t, mode); fi = np.array([j for j, r in enumerate(TGT) if r not in ex])
            if len(fi) >= 5: out.append((i, fi))
        _FOLDS[mode] = out
    return _FOLDS[mode]
YT = yday[TGT].values; QT = Q47[TGT].values

def L_gain(f, mode):
    """PP4 L pseudo-eval with a feature vector aligned to TGT; returns RMSE(M) - RMSE(M+a+b f)."""
    f = np.asarray(f, float); P = []; I = []
    for i, fi in folds(mode):
        res = YT[fi] - QT[fi]; x = f[fi]
        med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
        b, a = fit_lin(np.where(np.isfinite(x), x, med), res)
        ft = f[i] if np.isfinite(f[i]) else med
        P.append(QT[i] + a + b * ft); I.append(i)
    I = np.array(I); return rmse(QT[I], YT[I]) - rmse(np.array(P), YT[I])

if __name__ == "__main__":
    Ftg = F23.loc[TGT, FEATS].copy(); Ftg["AHxc"] = AHXc[TGT]
    segs = Cal.loc[TGT, "seg"].values; useg = np.unique(segs); gidx = [np.where(segs == u)[0] for u in useg]
    for mode in ["LOSO+", "BUF"]:
        real = {c: L_gain(Ftg[c].values, mode) for c in FEATS + ["AHxc"]}
        simp = {c: L_gain(SIMPLE.loc[TGT, c].values, mode) for c in SIMPLE.columns}
        print(f"\n=== {mode} seed 47: real gains (RMSE M − RMSE M+f), top 6:",
              {k: round(v, 4) for k, v in sorted(real.items(), key=lambda kv: -kv[1])[:6]})
        print("    simple summaries:", {k: round(v, 4) for k, v in simp.items()})
        Fm = Ftg[FEATS].values.astype(float); nA, nB = [], []
        for it in range(300):
            pi = rng.permutation(len(TGT)); Fp = Fm[pi]
            nA.append(max(L_gain(Fp[:, j], mode) for j in range(len(FEATS))))
            Fq = np.empty_like(Fm); perm = rng.permutation(len(useg))
            for gi, src in zip(gidx, perm):
                s_ = gidx[src]; Fq[gi] = Fm[s_[np.arange(len(gi)) % len(s_)]]
            nB.append(max(L_gain(Fq[:, j], mode) for j in range(len(FEATS))))
        print(f"    null95 of max gain over 21: record-perm {np.quantile(nA,.95):.4f}, seg-block-perm {np.quantile(nB,.95):.4f} | AHx real {real['AHx']:.4f} -> p_record {np.mean(np.array(nA)>=real['AHx']):.3f}, p_block {np.mean(np.array(nB)>=real['AHx']):.3f}")
    # (3) influence
    D = run_L("DIAG10", 47, "LOSO+", TGT)
    D["gain_sq"] = (D.M - D.y) ** 2 - (D.M_AHxc - D.y) ** 2
    print("\n(3) LOSO+ s47 top-5 records by squared-error reduction from AHxc:")
    print(D.sort_values("gain_sq", ascending=False).head(5)[["t", "seg", "y", "M", "M_AHxc", "gain_sq"]].round(3).to_string(index=False))
    print("    bottom-3 (most hurt):"); print(D.sort_values("gain_sq").head(3)[["t", "seg", "y", "M", "M_AHxc", "gain_sq"]].round(3).to_string(index=False))
    big = D.assign(ae=(D.y - D.M).abs()).sort_values("ae", ascending=False).t.head(3).tolist()
    S = D[~D.t.isin(big)]
    print(f"    without top-3 |y-q| records {big}: n {len(S)} RMSE M {rmse(S.M.values,S.y.values):.4f} M_AHxc {rmse(S.M_AHxc.values,S.y.values):.4f} M_icpt {rmse(S.M_icpt.values,S.y.values):.4f}")
