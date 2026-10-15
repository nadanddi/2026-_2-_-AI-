# -*- coding: utf-8 -*-
"""V3LF10 (2026-10-15 집 클로드; V3LF(6.929) 가 링크 30% 규칙에서 EXT10 안전 FAIL → 원래 V3L 규칙인 일률 10% 로 같은 저장 예측 재판정).
사전 고정(커밋 후 실행).  온도.  학습 없음(v3lf_ckpt 재사용).  한계: EXT10·EXT12 는 V3LF 에서 이미 본 판 — 10% 는 새로 고른 값이 아니라 V3L 원 규칙(6.912)·VW3(비율↑ → 추운 손해↑).
[아래는 V3LF 원 설명]
V3L(6.912, VW4) 의 링크를 현장형 링커 FL(fl3_train_links.csv, p ≥ .5; 번호 차이·날짜 추정 특징 없음)로 바꾼 V3LF.
학습·검증 시퀀스 모두 같은 FL 학습 표(대상 같은 농장 ±1 숨김, 평가 후보 제외)로 전날·전전날 창.  나머지 = V3L 과 같음.
배치: F·G·H·I, DIAG10·EL1·EXT10·EXT12, J·K·L.  섞기: FL 링크 있는 기록 30%, 없는 기록 10%(VW4 규칙).
판정(α .025): v23(V3 없음) 대비 DIAG10·EL1·F·G·H·I·J·K·L 모든 셀 개선 & 합친(DIAG10+F~L, 셀 1) 5일 블록 P(악화) < .025 & EXT10·EXT12 안전.
서술: V3L(v15R 링크, 30%-링크 규칙) 대비.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u v3lf10_flat10_v1.py sum
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CK = os.path.join(env.LOCAL, "v3lf_ckpt"); KINDS = ("V3LF",); CKL = os.path.join(env.LOCAL, "v3l_ckpt")
def link_maps():
    F = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); return {"V3LF": {r: p for r, p, q in zip(F.rec, F.prev, F.p) if q >= .5 and p != r}}
def sequences(lab, a, USE, link):
    a = a.copy(); a["rec"] = a.row_id.str[:7]; a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    cols = USE + ["hs", "hc"]; DAY = {}
    for r, g in a.groupby("rec"):
        m = np.full((24, len(cols)), np.nan, np.float32); g = g.sort_values("hour"); m[g.hour.values.astype(int)] = g[cols].to_numpy(np.float32); DAY[r] = m
    nanday = np.full((24, len(cols)), np.nan, np.float32)
    S = np.full((len(lab), 48, len(cols)), np.nan, np.float32)
    for j, (r, h) in enumerate(zip(lab.row_id.str[:7], lab.hour.astype(int))):
        p1 = link.get(r); p2 = link.get(p1) if p1 else None
        seq = np.concatenate([DAY.get(p2, nanday) if p2 else nanday, DAY.get(p1, nanday) if p1 else nanday, DAY[r][:h + 1]])
        S[j] = seq[-48:]
    return S
def run():
    S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
    LM = link_maps(); SS = {k: sequences(lab, a, S6.USE, LM[k]) for k in KINDS}
    for k in KINDS: print(k, "links", len(LM[k]), "| rows with full 48h window", float(np.isfinite(SS[k][:, 0, 4]).mean()), flush=True)
    jobs = [(L, ff(lab), mk) for L, (ff, mk, *_) in S7.LAY.items()] + [(name, folds, None) for name, folds in sets]
    for L_, f_ in {"J": "tt24_gru_daygate_confirm_v1.py", "K": "tt26_v3_layoutK_v1.py", "L": "tt27_chain_corrector_layoutL_v1.py"}.items():
        mod = load("lay" + L_, f_); jobs.append((L_, mod.fresh_folds(lab), mod.masks))
    import common
    for L, folds, mk in jobs:
        for kf, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (L, kf))
            if os.path.exists(path): continue
            tm, vm = (mk(lab, fd) if mk is not None else common.split_mask(lab, fd))
            if not vm.sum(): continue
            out = pd.DataFrame({"row_id": lab.row_id[vm].values})
            for k in KINDS: out[k] = np.mean([S7.fit_predict(SS[k][tm], y[tm], wb[tm], SS[k][vm], s, dev, "V3") for s in (1, 2)], 0)
            out.to_csv(path, index=False); print("%s/%d done | alone %s" % (L, kf, " ".join("%s %.3f" % (k, np.sqrt(np.mean((out[k].values - y[vm]) ** 2))) for k in KINDS)), flush=True)
def summarize():
    src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3sum"}
    exec(src[:src.index("JUDGE = [")], g)
    T, B, C2, anch, LV, D1, CT, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["rm"]
    if "L" not in T.LAY: T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
    rng = np.random.default_rng(20270109); FLK = set(link_maps()["V3LF"]); Lm = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); MLK = {r for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .5}
    def ld(d, L, col):
        fs = [f for f in os.listdir(d) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(d, f)) for f in fs]).drop_duplicates("row_id")[["row_id", col]]
    JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I", "J", "K", "L"]; ok = True; diffs = []; safety = {}
    for L in JUDGE + ["EXT10", "EXT12"]:
        d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; rec = (M.farm + "_" + M.day.map("{:03d}".format)).values
        X = M[["row_id"]].merge(ld(CK, L, "V3LF"), on="row_id", how="left").merge(ld(CKL, L, "V3L"), on="row_id", how="left")
        g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
        mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
        key = (M.farm + "_" + (M.day // 5).astype(str)).values; P = {"NO": [], "F": [], "V": []}
        wF = np.full(len(rec), .1); wV = np.where(np.isin(rec, list(MLK)), .3, .1)
        for bi, b in enumerate(bs):
            for fam in "AB":
                a = B.w40(M, b, cx, fam); P["NO"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV))
                for k, v, w in (("F", X.V3LF.values, wF), ("V", X.V3L.values, wV)):
                    miss = np.isnan(v); P[k].append(anch(M, C2.corrected(M, a + w * g_ * h_ * (~miss) * (np.nan_to_num(v) - a), D1, bi, fam, 7), LV))
        ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["F"], P["NO"])]; chv = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["F"], P["V"])]
        print(f"{L:6s} V3LF vs no-GRU {' '.join('%+.2f' % (100 * x) for x in ch)} | vs V3L(v15R) {np.mean(chv) * 100:+.2f}", flush=True)
        if L in JUDGE:
            ok &= all(x < 0 for x in ch)
            if L != "EL1": diffs.append(pd.Series((P["F"][0] - y) ** 2 - (P["NO"][0] - y) ** 2).groupby(key).mean())
        else:
            pw = []
            for n, b_ in zip(P["F"], P["NO"]):
                dd = pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
            safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok &= safety[L]
    S = pd.concat(diffs, axis=1).mean(axis=1).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
    print(f"pooled P(worse) {P_:.4f} | safety {safety}"); print("V3LF10:", "ADOPT" if (ok and P_ < .025) else "FAIL")
if __name__ == "__main__":
    run() if MODE == "cv" else summarize()
