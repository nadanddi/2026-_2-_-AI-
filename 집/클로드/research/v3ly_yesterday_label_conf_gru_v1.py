# -*- coding: utf-8 -*-
"""V3LY (2026-10-15 집 클로드; 사용자: 'V3L 에 1번(어제 실제 배지온도 곡선) 2번(링커 확신도) 묶어서 진행').  사전 고정(커밋 후 실행).  온도.
V3L(6.912 ADOPT): 2층 GRU 48시간 창의 앞부분 = 링커 v15R 거울 prev0(p0 ≥ .5)의 같은 동 전날·전전날 '입력'.
V3LY = V3L 에 채널 3개 추가 + 연결 문턱을 .3 으로 낮춤(확신도는 채널로 넘겨 GRU 가 스스로 깎게):
  LBL  = 그 위치 기록의 실제 배지온도(sub_temp 정답) — 그 폴드에서 '학습 행으로 보이는' 기록만(검증·숨김 기록 정답은 NaN), 오늘 위치는 언제나 NaN
  HAS  = LBL 이 있으면 1, 없으면 0
  CONF = 그 위치가 전날이면 p0(대상→전날), 전전날이면 p0(대상→전날)·p0(전날→전전날), 오늘 위치 1, 연결 없음 0
  나머지(USE 14 + 시각 sin·cos, 표준화, 결측 표시, Adam 2e-3, 25 에폭, 배치 256, 가중 MSE, 시드 1·2) V3L 과 같음.  LBL 은 표준화 후 NaN → 0.
배치: V3L 과 같음(F·G·H·I, DIAG10·EL1·EXT10·EXT12).  V3L 저장 예측(local/v3l_ckpt) 을 기준으로.
판정(단일 안 α .025, 비교 대상 = V3L):  v23 위 10% 섞기(날 문턱 그대로)에서 V3LY 가 V3L 보다
  DIAG10·EL1·F·G·H·I 모든 셀 개선 & 합친(DIAG10+F+G+H+I, 셀 1) 5일 블록 P(V3LY 악화) < .025 & EXT10·EXT12 안전(V3L 대비: 셀 평균 악화 & '나음' 비율 < .025 이면 FAIL).
서술: V3LY 단독 RMSE, v23(GRU 없음) 대비 %, 2차·추운 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u v3ly_yesterday_label_conf_gru_v1.py {cv|sum}
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CK = os.path.join(env.LOCAL, "v3ly_ckpt"); CKL = os.path.join(env.LOCAL, "v3l_ckpt"); PTH = .3
def link_map():
    PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv"))
    return {r: (p, float(q)) for r, p, q in zip(PM.rec, PM.prev0, PM.p0) if isinstance(p, str) and q >= PTH and p != r}
def base_arrays(lab, a, USE, link):
    """input sequences (N, 48, F) + per-position record index (−1 none) + per-position conf + hour."""
    a = a.copy(); a["rec"] = a.row_id.str[:7]; a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    cols = USE + ["hs", "hc"]; recs = sorted(a.rec.unique()); ri = {r: i for i, r in enumerate(recs)}
    DAY = np.full((len(recs), 24, len(cols)), np.nan, np.float32)
    for r, g in a.groupby("rec"): DAY[ri[r], g.hour.values.astype(int)] = g[cols].to_numpy(np.float32)
    n = len(lab); S = np.full((n, 48, len(cols)), np.nan, np.float32); PR = np.full((n, 48), -1, np.int32); PC = np.zeros((n, 48), np.float32); PH = np.zeros((n, 48), np.int32)
    for j, (r, h) in enumerate(zip(lab.row_id.str[:7], lab.hour.astype(int))):
        l1 = link.get(r); p1, c1 = (l1 if l1 else (None, 0.0)); l2 = link.get(p1) if p1 else None; p2, c2 = (l2 if l2 else (None, 0.0))
        if p2 == r: p2, c2 = None, 0.0
        segs = [(p2, c1 * c2, 24), (p1, c1, 24), (r, 1.0, h + 1)]; buf, rr, cc, hh = [], [], [], []
        for rec, cf, L in segs:
            buf.append(DAY[ri[rec], :L] if rec in ri else np.full((L, len(cols)), np.nan, np.float32))
            rr += [ri[rec] if rec in ri else -1] * L; cc += [cf if rec in ri else 0.0] * L; hh += list(range(L))
        seq = np.concatenate(buf); S[j] = seq[-48:]; PR[j] = np.array(rr[-48:]); PC[j] = np.array(cc[-48:]); PH[j] = np.array(hh[-48:])
        PR[j, -(h + 1):] = -2   # today positions: never carry a label
    return S, PR, PC, PH, recs
def with_labels(S, PR, PC, PH, LBL, visible):
    """append LBL, HAS, CONF channels; LBL only for records visible as training in this fold."""
    ok = (PR >= 0) & visible[np.clip(PR, 0, None)]
    lb = np.where(ok, LBL[np.clip(PR, 0, None), PH], np.nan).astype(np.float32); has = np.isfinite(lb).astype(np.float32)
    return np.concatenate([S, lb[..., None], has[..., None], PC[..., None]], 2)
def run():
    S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W
    import env_extra_gpu  # noqa: F401
    import torch
    import common
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
    link = link_map(); S, PR, PC, PH, recs = base_arrays(lab, a, S6.USE, link); ri = {r: i for i, r in enumerate(recs)}
    LBL = np.full((len(recs), 24), np.nan, np.float32)
    for r, g in lab.assign(rec=lab.row_id.str[:7]).groupby("rec"): LBL[ri[r], g.hour.values.astype(int)] = g.sub_temp.values
    print("links (p0>=%.1f)" % PTH, len(link), "| rows with a linked yesterday", float((PR[:, 23] >= 0).mean() if False else (PR >= 0).any(1).mean()), flush=True)
    rec_of_row = np.array([ri[r] for r in lab.row_id.str[:7]])
    jobs = [(L, ff(lab), mk) for L, (ff, mk, *_) in S7.LAY.items()] + [(name, folds, None) for name, folds in sets]
    for L, folds, mk in jobs:
        for kf, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (L, kf))
            if os.path.exists(path): continue
            tm, vm = (mk(lab, fd) if mk is not None else common.split_mask(lab, fd))
            if not vm.sum(): continue
            visible = np.zeros(len(recs), bool); visible[np.unique(rec_of_row[tm])] = True
            SX = with_labels(S, PR, PC, PH, LBL, visible)
            out = pd.DataFrame({"row_id": lab.row_id[vm].values})
            out["V3LY"] = np.mean([S7.fit_predict(SX[tm], y[tm], wb[tm], SX[vm], s, dev, "V3") for s in (1, 2)], 0)
            out.to_csv(path, index=False); print("%s/%d done | alone V3LY %.3f | val rows with yesterday label %.2f" % (L, kf, np.sqrt(np.mean((out.V3LY.values - y[vm]) ** 2)), float(np.isfinite(SX[vm][:, :, -3]).any(1).mean())), flush=True)
def summarize():
    src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3sum"}
    exec(src[:src.index("JUDGE = [")], g)
    T, B, C2, anch, LV, D1, CT, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["rm"]
    rng = np.random.default_rng(20261227)
    def ld(d, L):
        fs = [f for f in os.listdir(d) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(d, f)) for f in fs]).drop_duplicates("row_id")
    JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I"]; ok = True; diffs = []; safety = {}
    for L in JUDGE + ["EXT10", "EXT12"]:
        d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
        X = M[["row_id"]].merge(ld(CKL, L)[["row_id", "V3L"]], on="row_id", how="left").merge(ld(CK, L), on="row_id", how="left")
        g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
        mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
        key = (M.farm + "_" + (M.day // 5).astype(str)).values; p2 = M.day.values >= 179; cold = M.in_temp.values <= 8; nog, ref, new = [], [], []
        for bi, b in enumerate(bs):
            for fam in "AB":
                a = B.w40(M, b, cx, fam); nog.append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV)); out = {}
                for k in ("V3L", "V3LY"):
                    v = X[k].values; miss = np.isnan(v); w = .1 * g_ * h_ * (~miss); out[k] = anch(M, C2.corrected(M, a + w * (np.nan_to_num(v) - a), D1, bi, fam, 7), LV)
                ref.append(out["V3L"]); new.append(out["V3LY"])
        ch = [rmf(n - y) / rmf(r_ - y) - 1 for n, r_ in zip(new, ref)]; chn = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(new, nog)]
        print(f"{L:6s} V3LY vs V3L all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * (rmf((n - y)[p2]) / rmf((r_ - y)[p2]) - 1)) for n, r_ in zip(new, ref)) if p2.any() else '-'}"
              f" | cold {' '.join('%+.2f' % (100 * (rmf((n - y)[cold]) / rmf((r_ - y)[cold]) - 1)) for n, r_ in zip(new, ref)) if cold.any() else '-'} | vs no-GRU {np.mean(chn) * 100:+.2f} | alone V3LY {rmf(np.nan_to_num(X.V3LY.values, nan=np.nanmean(X.V3LY)) - y):.3f} V3L {rmf(np.nan_to_num(X.V3L.values, nan=np.nanmean(X.V3L)) - y):.3f}", flush=True)
        if L in JUDGE:
            ok &= all(x < 0 for x in ch)
            if L != "EL1": diffs.append(pd.Series((new[0] - y) ** 2 - (ref[0] - y) ** 2).groupby(key).mean())
        else:
            pw = []
            for n, r_ in zip(new, ref):
                dd = pd.Series((n - y) ** 2 - (r_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
            safety[L] = not (np.mean(ch) > 0 and min(pw) < .025); ok &= safety[L]
    S = pd.concat(diffs, axis=1).mean(axis=1).values; P = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
    print(f"\npooled P(V3LY worse than V3L) {P:.4f} | safety {safety}\nV3LY:", "ADOPT (replaces V3L)" if (ok and P < .025) else "FAIL")
if __name__ == "__main__":
    run() if MODE == "cv" else summarize()
