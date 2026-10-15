# -*- coding: utf-8 -*-
"""YL1 (2026-10-15 집 클로드; 사용자 '온도 0.44 까지' → 안 B).  사전 고정(커밋 후 실행).  온도.
진단(커밋 전 서술): 기본 모델의 기록 간 특징(_pd1·r72m·r168m·ph_*24/72·lag24)은 '번호상 앞 기록'으로 계산되는데, 앞 번호 기록과 0시 외기 차이
  중앙값이 학습 1차 .60℃ → 학습 2차 1.25℃·평가(전부 2차) 1.10℃ (FL 링크 어제는 .40/.55/.55) → 평가 행에서 '어제' 특징의 뜻이 학습과 다름.
안 B: 기존 특징은 그대로 두고, FL 링커가 찾은 같은 동 어제(p ≥ .5; 학습표 fl3_train_links — 대상 같은 농장 ±1 정답 숨김·평가 후보 제외, 입력만)의
  요약을 새 열로 넣음(못 찾으면 빈칸 + 표시 0):
  y_link, y_p, y_in_mean, y_in_min, y_in_max, y_in_23, y_out_mean, y_out_23, y_rad_sum, y_heat_mean, y_thermal_mean, d_in0 (= 오늘 0시 실내 − 어제 23시 실내)
  BASE(ct)·CODEX(FEATURE_COLUMNS)·TabPFN(X) 모두에 추가. 나머지(구성원·가중·시드 BASE 1616/1717·CODEX 726·PFN 73~80) = OS1 v2 와 같음.
배치: N·CK2(기준 = OS1 v2 저장 셀 os1_base_ckpt, 같은 폴드·시드), EL1(2차 5일 묶음 — 평가 닮음; 기준과 후보를 여기서 둘 다 학습, common.split_mask).
판정(단일 안 α .025): YL 셀 vs 기준 셀(W40G-S 혼합 w40, 앵커 없이) — N·CK2·EL1 12셀 모두 개선 & 세 배치 합친(배치 x 농장 x 5일 블록) P(악화) < .025.
서술: 배치별 2차·추운 행, K8 앵커 + TX3 를 얹은 경우.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u yl1_linked_yesterday_features_v1.py {fit|sum}
"""
import os, sys, gc, pathlib, importlib.util, pickle
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "sum"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
O1 = load("os1v2", "os1_temp_regime_split_v2.py"); CKY = os.path.join(env.LOCAL, "yl1_ckpt"); CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
YC = ["y_link", "y_p", "y_in_mean", "y_in_min", "y_in_max", "y_in_23", "y_out_mean", "y_out_23", "y_rad_sum", "y_heat_mean", "y_thermal_mean", "d_in0"]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))

def yfeat(row_ids, links, X):
    """row_ids: hourly rows; links: rec -> (prev, p); X: raw inputs (train_X ∪ usable test_X) indexed by row_id."""
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int)
    P = {c: X.pivot(index="rec", columns="h", values=c) for c in ("in_temp", "out_temp", "out_rad", "act_heating", "act_thermal")}
    R = {}
    for r in set(s[:7] for s in row_ids):
        d = dict.fromkeys(YC, np.nan); d["y_link"] = 0.0; d["y_p"] = 0.0
        pv, p = links.get(r, (None, 0.0))
        if isinstance(pv, str) and p >= .5 and pv in P["in_temp"].index:
            ti = P["in_temp"].loc[pv]; to = P["out_temp"].loc[pv]
            d.update(y_link=1.0, y_p=float(p), y_in_mean=ti.mean(), y_in_min=ti.min(), y_in_max=ti.max(), y_in_23=ti.get(23, np.nan), y_out_mean=to.mean(), y_out_23=to.get(23, np.nan),
                     y_rad_sum=P["out_rad"].loc[pv].sum(min_count=1), y_heat_mean=P["act_heating"].loc[pv].mean(), y_thermal_mean=P["act_thermal"].loc[pv].mean())
            t0 = P["in_temp"].loc[r].get(0, np.nan) if r in P["in_temp"].index else np.nan; d["d_in0"] = t0 - d["y_in_23"]
        R[r] = d
    return pd.DataFrame([R[s[:7]] for s in row_ids], columns=YC)

def run_fit():
    T27 = O1.load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W; W.OUT = pathlib.Path(env.LOCAL) / "yl1_season"
    os.makedirs(CKY, exist_ok=True); os.makedirs(CKR, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    tX, ty, sX = W.TM.masked_loader(); FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); LK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
    YF = yfeat(lab.row_id.values, LK, tX); assert len(YF) == len(lab)
    for c in YC: lab[c] = YF[c].values; pfn[c] = YF[c].values
    print("linked records share:", float(lab.drop_duplicates(["farm", "day"]).y_link.mean()), flush=True)
    import common
    FC0 = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    el = dict(sets)["EL1"]; LAYS = [("N", O1.layout_folds(W, lab)["N"], O1.masks, (CKY,)), ("CK2", O1.layout_folds(W, lab)["CK2"], O1.masks, (CKY,)),
                                     ("EL1", el, common.split_mask, (CKY, CKR))]
    for L, folds, mk, dirs in LAYS:
        for k, fd in enumerate(folds):
            tm, vm = mk(lab, fd)
            if not vm.sum(): continue
            for D in dirs:
                path = os.path.join(D, "%s_%d.csv" % (L, k))
                if os.path.exists(path): continue
                use = D == CKY; ctx = ct + YC if use else ct; FC = FC0 + YC if use else FC0; XC = list(W.FEATURE_COLUMNS) + (YC if use else [])
                out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); out["fold"] = k
                tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "yl1_%s" % L, k)
                for s in O1.SEEDS: out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ctx, phc, s)
                save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = FC
                try: out["codex_R"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
                finally: W.TM.FEATURE_COLUMNS = save
                X = pfn[XC].to_numpy(np.float32); XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season; ps = []
                for seed in range(73, 81):
                    idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
                    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed, ignore_pretraining_limits=True, inference_precision=torch.float32)
                    m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
                out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0); out.to_csv(path, index=False)
                print("%s %s %d done (val %d)" % ("YL" if use else "REF", L, k, vm.sum()), flush=True)

def summarize():
    B = O1.load("tc1b", "tc1b_chain_history_common_train_v1.py"); RD = O1.RD; rng = np.random.default_rng(20270123)
    def frame(D, L):
        fs = [f for f in sorted(os.listdir(D)) if f.startswith(L + "_")]; M = pd.concat([pd.read_csv(os.path.join(D, f)) for f in fs], ignore_index=True)
        return M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].set_index("row_id")
    ok = True; diffs = []
    for L, refdir in (("N", O1.CKB), ("CK2", O1.CKB), ("EL1", CKR)):
        Y = frame(CKY, L); R = frame(refdir, L).reindex(Y.index); y = Y.sub_temp.values
        cells = [(B.w40(Y.reset_index(), "base_R_%d" % s, "codex_R", fam), B.w40(R.reset_index(), "base_R_%d" % s, "codex_R", fam)) for s in O1.SEEDS for fam in "AB"]
        ch = [rm(a - y) / rm(b - y) - 1 for a, b in cells]; ok &= all(x < 0 for x in ch)
        key = (L + "_" + Y.farm + "_" + (Y.day // 5).astype(str)).values
        diffs.append(pd.Series(np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in cells], 0)).groupby(key).mean())
        p2 = Y.day.values >= 179; cold = (Y.in_temp <= 8).values
        sub = lambda m: 100 * np.mean([rm((a - y)[m]) / rm((b - y)[m]) - 1 for a, b in cells]) if m.any() else np.nan
        print(f"{L:3s} YL vs REF {' '.join('%+.3f' % (100 * x) for x in ch)} | p2 {sub(p2):+.2f}% cold {sub(cold):+.2f}% | RMSE REF {np.mean([rm(b - y) for a, b in cells]):.4f} YL {np.mean([rm(a - y) for a, b in cells]):.4f} (rows {len(y)})", flush=True)
    S = pd.concat(diffs).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
    print(f"pooled P(worse) {P_:.4f} | all 12 cells better {ok} -> YL1: {'ADOPT' if ok and P_ < .025 else 'FAIL'}")

if __name__ == "__main__":
    {"fit": run_fit, "sum": summarize}[MODE]()
