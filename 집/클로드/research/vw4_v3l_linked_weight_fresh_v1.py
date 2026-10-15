# -*- coding: utf-8 -*-
"""VW4 (2026-10-15 집 클로드; 사용자 '온도도 EC 처럼' → RVT(사후)에서 'V3L 을 링크 있는 날만 30%' 가 6판 모두 개선).  사전 고정(커밋 후 실행).  온도.
RVT 에서 고른 후보를, 고를 때 안 쓴 새 배치 J·K·L(VW3 에서 학습해 둔 V3L 체크포인트)에서 판정.
  W3L = 링크(패키지 25 MASK 거울 링커 p0 ≥ .5) 있는 기록 V3L 30%, 없는 기록 10% (날 문턱 g·h 그대로)  vs  W1 = V3L 10%.
판정(α .025): J·K·L 모든 셀 W3L < W1 & 합친(J+K+L, 셀 1) 5일 블록 P(악화) < .025 & EXT10·EXT12 안전(W3L vs W1: 셀 평균 악화 & '나음' 비율 < .025 이면 FAIL).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u vw4_v3l_linked_weight_fresh_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
V3LM = load("v3l", "v3l_linked_yesterday_gru_v1.py"); CK = os.path.join(env.LOCAL, "v3l_ckpt")
Lm = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); MLINK = {r for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .5}
NEW = {"J": "tt24_gru_daygate_confirm_v1.py", "K": "tt26_v3_layoutK_v1.py", "L": "tt27_chain_corrector_layoutL_v1.py"}
def run():
    S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
    S = V3LM.sequences(lab, a, S6.USE, V3LM.link_maps()["V3L"])
    for L, f in NEW.items():
        mod = load("lay" + L, f)
        for kf, fd in enumerate(mod.fresh_folds(lab)):
            path = os.path.join(CK, "%s_%d.csv" % (L, kf))
            if os.path.exists(path): continue
            tm, vm = mod.masks(lab, fd)
            if not vm.sum(): continue
            out = pd.DataFrame({"row_id": lab.row_id[vm].values}); out["V3L"] = np.mean([S7.fit_predict(S[tm], y[tm], wb[tm], S[vm], s, dev, "V3") for s in (1, 2)], 0)
            out.to_csv(path, index=False); print("%s/%d done | alone V3L %.3f" % (L, kf, np.sqrt(np.mean((out.V3L.values - y[vm]) ** 2))), flush=True)
def summarize():
    src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3vw3"}
    exec(src[:src.index("JUDGE = [")], g)
    T, B, C2, anch, LV, D1, CT, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["rm"]
    if "L" not in T.LAY: T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
    rng = np.random.default_rng(20261231)
    def ld(L):
        fs = [f for f in os.listdir(CK) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(CK, f)) for f in fs]).drop_duplicates("row_id")
    ok = True; diffs = []; safety = {}
    for L in ["J", "K", "L", "EXT10", "EXT12"]:
        d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
        v = M[["row_id"]].merge(ld(L)[["row_id", "V3L"]], on="row_id", how="left").V3L.values; miss = np.isnan(v)
        g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
        mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
        key = (M.farm + "_" + (M.day // 5).astype(str)).values; p2 = M.day.values >= 179; cold = M.in_temp.values <= 8; P = {"NO": [], "W1": [], "W3": []}
        for bi, b in enumerate(bs):
            for fam in "AB":
                a = B.w40(M, b, cx, fam); P["NO"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV))
                lkm = np.isin((M.farm + "_" + M.day.map("{:03d}".format)).values, list(MLINK))
                for k, w in (("W1", .1), ("W3", np.where(lkm, .3, .1))):
                    wt = w * g_ * h_ * (~miss); P[k].append(anch(M, C2.corrected(M, a + wt * (np.nan_to_num(v) - a), D1, bi, fam, 7), LV))
        ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["W3"], P["W1"])]; v1 = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["W1"], P["NO"])]
        f2 = lambda A, Bb, m: ' '.join('%+.2f' % (100 * (rmf((n - y)[m]) / rmf((b_ - y)[m]) - 1)) for n, b_ in zip(A, Bb)) if m.any() else '-'
        print(f"{L:6s} missing {int(miss.sum())} | W3 vs W1 all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {f2(P['W3'], P['W1'], p2)} | cold {f2(P['W3'], P['W1'], cold)} || V3L(W1) vs no-GRU all {' '.join('%+.2f' % (100 * x) for x in v1)}", flush=True)
        if L in ("J", "K", "L"):
            ok &= all(x < 0 for x in ch); diffs.append(pd.Series((P["W3"][0] - y) ** 2 - (P["W1"][0] - y) ** 2).groupby(key).mean())
        else:
            pw = []
            for n, b_ in zip(P["W3"], P["W1"]):
                dd = pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
            safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok &= safety[L]
    S = pd.concat(diffs, axis=1).mean(axis=1).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
    print(f"\npooled P(W3 worse) {P_:.4f} | safety {safety}\nVW4:", "ADOPT (linked 30%)" if (ok and P_ < .025) else "FAIL (keep .10/GH)")
if __name__ == "__main__":
    summarize()
