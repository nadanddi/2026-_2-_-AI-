# -*- coding: utf-8 -*-
"""SNF2 (2026-10-16 집 클로드; SNF1(7.005) 센서 이상 감지 -> 사용자 '진행해': 학습 정리 후 기본 모델 재학습·조립 판정).  사전 고정(커밋 후 실행).  온도.
학습 정리: snf1_flags_v1.csv 의 F2(값 멈춤)·F3(실내 온도 불일치) 학습 행(45행) 의 BASE·CODEX 행 가중(wb)·PFN 문맥 가중(wp)을 0 으로 (검증 대상 행 입력·정답은 그대로;
  10-10 규정 판단: 학습 정리는 비인과 허용).  F1(직선)은 해롭지 않아(7.005) 정리 안 함.
재학습: OS1 v2 / CV8 과 같은 코드(BASE 1616·1717, CODEX 726, TabPFN 73~80 GPU), 배치 N·CK2(OS1 폴드)·CV8(CV8 폴드) -> local/snf2_ckpt.
판정: 19회차 온도 구성(셀 -> FL K8 -> TX3 .25/10/.8 -> 시간 감쇠 사슬 M1, FL7 거울 표) 을 정리 전(기존 ckpt) vs 정리 후(새 ckpt) 로 조립해 비교.
  기준(단일 안, alpha .025): N·CK2·CV8 12셀 모두 개선 & 같은 날 중복 제거 P(악화) < .025.  서술: 정리 행이 있던 기록 vs 나머지, 추운 행.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u snf2_clean_train_rebase_v1.py {base|judge}
"""
import os, sys, gc, pathlib, importlib.util
MODE = sys.argv[1] if len(sys.argv) > 1 else "base"
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
CK = os.path.join(env.LOCAL, "snf2_ckpt"); SEEDS = (1616, 1717)
FLAG = pd.read_csv(os.path.join(HERE, "snf1_flags_v1.csv")); BAD = set(FLAG[(FLAG.set == "train") & (FLAG.F2 | FLAG.F3)].row_id)
def run_base():
    O1 = load("os1", "os1_temp_regime_split_v2.py"); C8 = load("cv8", "cv8_cold_validator_v1.py")
    T27 = load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W; W.OUT = pathlib.Path(env.LOCAL) / "snf2_season"
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb, wp = np.asarray(wb, float).copy(), np.asarray(wp, float).copy()
    rid_l = lab.row_id.values; rid_p = pfn.row_id.values if "row_id" in pfn else rid_l
    wb[np.isin(rid_l, list(BAD))] = 0.0; wp[np.isin(rid_p, list(BAD))] = 0.0
    print("zeroed rows: base", int(np.isin(rid_l, list(BAD)).sum()), "| pfn", int(np.isin(rid_p, list(BAD)).sum()), flush=True)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    LAYS = [("N", LF_, O1.masks) for LF_ in [O1.layout_folds(W, lab)["N"]]] + [("CK2", O1.layout_folds(W, lab)["CK2"], O1.masks), ("CV8", C8.layout_folds(W, lab)["CV8"], C8.masks)]
    for L, folds, mk in LAYS:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (L, k))
            if os.path.exists(path): continue
            tm, vm = mk(lab, fd)
            if not vm.sum(): continue
            out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); out["fold"] = k
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "snf2_%s" % L, k)
            for s in SEEDS: out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
            save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = FC
            try: out["codex_R"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
            finally: W.TM.FEATURE_COLUMNS = save
            XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season; ps = []
            pw = wp[tm] / wp[tm].sum()
            for seed in range(73, 81):
                idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int((pw > 0).sum())), replace=False, p=pw)
                m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed, ignore_pretraining_limits=True, inference_precision=torch.float32)
                m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
            out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
            out.to_csv(path, index=False); print("%s %d done" % (L, k), flush=True)
def judge():
    src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read(); G = {"__file__": os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), "__name__": "kf3snf2"}
    exec(src[:src.index("rng = np.random.default_rng(20270118)")], G)
    O1, B, W, lab, LF, RD, BETA, BIAS, IDX, ST, IN_ = (G[k] for k in ("O1", "B", "W", "lab", "LF", "RD", "BETA", "BIAS", "IDX", "ST", "IN_"))
    C8 = load("cv8j", "cv8_cold_validator_v1.py"); CM = G["CM"]
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    def tx3(M, key):
        cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]), nan=0)
        return .25 * np.clip((cm - .8) / .4, 0, 1) * np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0)
    MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); RES = {}
    for L, ckb, folds, mk in (("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("CV8", C8.CKB, C8.layout_folds(W, lab)["CV8"], C8.masks)):
        P = {}
        for tag, d in (("OLD", ckb), ("NEW", CK)):
            M = pd.concat([pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith(L + "_")], ignore_index=True)
            M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].sort_values("row_id").reset_index(drop=True)
            y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
            TXo = tx3(M, key); TRS, VF = {}, {}
            for k, fd in enumerate(folds):
                tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
                for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
            LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; AF, CHN = {}, {}
            for r in recs:
                if r not in LK or not isinstance(LK[r][0], str): continue
                p, q = LK[r]
                if p not in IDX or q < .8: continue
                dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
                if not np.isfinite(dt) or dt > 1.5: continue
                if r in TRS and p in TRS[r]:
                    if np.isfinite(ST[IDX[p], 23]): AF[r] = float(ST[IDX[p], 23])
                elif p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and VF.get(p) == VF.get(r) and (h[pos[p]] == 23).any(): CHN[r] = p
            cells = []
            for s in SEEDS:
                for fam in "AB":
                    x = B.w40(M, "base_R_%d" % s, "codex_R", fam); out = x.copy()
                    for r, a in AF.items():
                        ix = pos[r]; x0 = x[ix][h[ix] == 0]
                        if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
                    out = out + TXo
                    for r in sorted(recs, key=lambda r: (r[:3], int(r[4:7]))):
                        if r not in CHN: continue
                        ix = pos[r]; ip = pos[CHN[r]]; x0 = x[ix][h[ix] == 0]; a = out[ip][h[ip] == 23]
                        if len(x0) and len(a): out[ix] += .9 * np.exp(-h[ix] / 8.0) * BETA[h[ix]] * np.clip(a[0] - BIAS - x0[0], -2, 2)
                    cells.append(out)
            P[tag] = cells; P["y"] = y; P["rid"] = M.row_id.values; P["t"] = M.in_temp.values; P["rec"] = key; P["bk"] = (M.farm + "_" + (M.day // 5).astype(str)).values
        RES[L] = P; print(L, "assembled", flush=True)
    badrec = {r[:7] for r in BAD}
    def cellch(L, m=None):
        P = RES[L]; y = P["y"]; m = np.ones(len(y), bool) if m is None else m
        return [rm((a - y)[m]) / rm((b - y)[m]) - 1 for a, b in zip(P["NEW"], P["OLD"])]
    for L in RES:
        P = RES[L]; br = np.isin(P["rec"], list(badrec)); tt = np.nan_to_num(P["t"], nan=99)
        print(f"  NEW vs OLD | {L}: cells {' '.join('%+.3f' % (100 * x) for x in cellch(L))} | records that had cleaned rows {100 * np.mean(cellch(L, br)) if br.any() else float('nan'):+.2f}% | others {100 * np.mean(cellch(L, ~br)):+.2f}% | <=8C {100 * np.mean(cellch(L, tt <= 8)):+.2f}%")
    rng = np.random.default_rng(20270163)
    per = {L: pd.Series(np.mean([(a - RES[L]["y"]) ** 2 - (b - RES[L]["y"]) ** 2 for a, b in zip(RES[L]["NEW"], RES[L]["OLD"])], 0)).groupby(RES[L]["bk"]).mean() for L in RES}
    U = pd.concat(per, axis=1).mean(axis=1).values; pw = float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()); cells = sum((cellch(L) for L in RES), [])
    print(f"SNF2 cleaned-train rebase on round-19 temperature: 12 cells all better {all(x < 0 for x in cells)} | dedup P(worse) {pw:.4f} -> {'PASS' if all(x < 0 for x in cells) and pw < .025 else 'FAIL'}")
if __name__ == "__main__":
    {"base": run_base, "judge": judge}[MODE]()
