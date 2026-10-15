# -*- coding: utf-8 -*-
"""CV8 (2026-10-15 집 클로드; DV1 6.960: 평가가 학습보다 훨씬 추움(하루 최저 ≤ 6℃ 기록 평가 13/60 vs 학습 9/400) → 평가형 추운 검증 배치).
예측 생성 전용(판정 없음): 학습 기록 중 하루 최저 실내 ≤ 8℃ 인 42기록을 농장별 날짜 순으로 6폴드(i·6//n)로 빼고 나머지로 학습(같은 농장 ±1·다른 농장 ±5 제외).
구성원·시드는 OS1 v2 와 같음(BASE 1616/1717, CODEX 726, PFN 73~80, V3LF 시드 3·4). 결과 → local/cv8_base_ckpt, local/cv8_gru_ckpt (EL1·N 등은 기존 ckpt 재사용).
[아래는 OS1 v2 원 설명]
OS1 v2 (2026-10-15 — v1 은 CK 마지막 폴드(이어진 추운 묶음)에서 한 농장의 2차 학습 날이 0 이 되어 계절 지표 단계가 실패(결과 보기 전 설계 오류).
  v2: CK → CK2 = 같은 추운 날을 4일 블록으로 번갈아 5폴드(fold = (i // 4) % 5). N 은 그대로(저장 폴드 재사용). 나머지·판정 규칙 동일.
[아래는 v1 설명]
OS1 (2026-10-15 집 클로드; 사용자: '특정 부분에서만 빛난 실험 떼어 쓰기' 2번 — 실내온도로 따뜻한/추운 행 나눠 쓰기).  사전 고정(커밋 후 실행).  온도.
배경: V3LF(6.929·6.930) 따뜻한 판 −0.7~−2.0% 이나 추운 외삽 EXT10 손해(+.15~+.52%); AT1 v2(6.928) 낮은 확신 FL 앵커는 추운 행만 9판 −1.1~−1.5%.
  둘 다 이미 본 판에서 찾은 구간 → 처음 쓰는 배치 2개로 판정.
새 배치(둘 다 처음 사용, 사용자 노트북 GPU):
  N  : 농장별 학습 날 정렬, 6일 묶음, fold = (i // 6 + 3) % 10  (DIAG10·F~L 과 다른 나눔).  학습 제외 = 같은 농장 ±1, 다른 농장 ±5.
  CK : 추운 날(EXT 정의와 같은 FULL ph_in_temp_3 하루 최저 < 12℃) 을 농장별 정렬해 이어진 5묶음(fold = i·5 // n). 나머지 날(다른 추운 날 포함)로 학습, 같은 제외 규칙.
       (EXT10·12 는 추운 날 전부를 한 번에 빼 따뜻한 날로만 학습 — CK 는 다른 추운 날이 학습에 있는 평가 닮은 꼴)
  기본 구성원(TT27 과 같은 코드): BASE 시드 1616·1717, CODEX 726, TabPFN 계절 문맥 시드 73~80(A 73~76, B 77~80).  V3LF(FL 학습표 p ≥ .5 창) 시드 3·4.
기준 BASE27 = W40G-S 셀(사슬 없음) → FL 어제 온도 앵커 K8(fl2_cv_FL, p ≥ .8·정답 전날·|dT| ≤ 1.5, TA8 β, 편향 .36, 자르기 2) → TX3(6.931).
  앵커 전날은 그 배치 폴드의 학습 쪽 기록일 때만(검증·제외 구역 기록의 정답은 안 씀) — 모든 안에 같게.
안(미리 고정, 새로 맞춘 값 없음):
  C1 GRU 엄격 게이트: q' = q + .10 · g' · h' · (V3LF − q),  g' = clip((in_temp − 10)/2, 0, 1)(결측 1), h' = clip((그날 0..h 실내 최저 − 12)/2, 0, 1)(결측 1)
     — V3L 원 게이트(8℃·10℃ 시작)를 램프 한 폭(2℃)씩 올림.  q' → K8 → TX3.
  C2 추운 행 앵커 넓히기: BASE27 + c_h · p · β_h · clip(A23 − .36 − q0, ±2)  (FL p ∈ [.5, .8), 정답 전날·학습 쪽·|dT| ≤ 1.5 인 기록만),  c_h = clip((10 − in_temp_h)/2, 0, 1)(결측 0).
  서술: C0 = V3LF 원 게이트 10%, C1+C2.
판정(본페로니 k 2 → α .0125, 거친 날 제외, 블록 = 농장 x 5일):
  C1: N 4셀 모두 BASE27 보다 개선 & N P(악화) < .0125 & CK 안전(셀 평균 악화 & P(나음) < .0125 이면 FAIL).
  C2: CK 4셀 모두 개선 & CK P(악화) < .0125 & N 안전.
  서술: EXT10·EXT12·DIAG10 의 저장 셀(사슬 없음) 위 같은 계산(이미 본 판이라 판정 아님), 추운 행·2차 분해.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cv8_cold_validator_v1.py {base|gru}
"""
import os, sys, gc, pathlib, importlib.util, pickle
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "sum"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CKB = os.path.join(env.LOCAL, "cv8_base_ckpt"); CKG = os.path.join(env.LOCAL, "cv8_gru_ckpt")
SEEDS = (1616, 1717); GSEEDS = (3, 4); LAYS = ("CV8",)
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))

def folds_N(lab):
    out = [{f: set() for f in ("F13", "F47")} for _ in range(10)]
    for f in ("F13", "F47"):
        for i, d in enumerate(sorted(lab[lab.farm == f].day.unique())): out[(i // 6 + 3) % 10][f].add(int(d))
    return out
def cold_days(W):
    labF, _, _ = W.TM.build_world(); dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min(); return {(f, int(d)) for (f, d), v in dmin.items() if v < 12}
def folds_CK(lab, CD):
    out = [{f: set() for f in ("F13", "F47")} for _ in range(5)]
    for f in ("F13", "F47"):
        ds = sorted(int(d) for d in lab[lab.farm == f].day.unique() if (f, int(d)) in CD); n = len(ds)
        for i, d in enumerate(ds): out[(i // 4) % 5][f].add(d)
    return out
def masks(lab, fd):
    val = np.zeros(len(lab), bool); bad = np.zeros(len(lab), bool); d = lab.day.values
    for f, ds in fd.items():
        if not ds: continue
        isf = (lab.farm == f).values; ds = np.array(sorted(ds))
        val |= isf & np.isin(d, ds)
        bad |= (isf & np.isin(d, np.unique((ds[:, None] + np.arange(-1, 2)).ravel()))) | (~isf & np.isin(d, np.unique((ds[:, None] + np.arange(-5, 6)).ravel())))
    return ~bad & ~val, val
def cold8_folds(lab):
    m = lab.groupby(["farm", "day"]).in_temp.min(); out = [{f: set() for f in ("F13", "F47")} for _ in range(6)]
    for f in ("F13", "F47"):
        ds = sorted(int(d) for (ff, d), v in m.items() if ff == f and v <= 8); n = len(ds)
        for i, d in enumerate(ds): out[i * 6 // n][f].add(d)
    return out
def layout_folds(W, lab):
    return {"CV8": cold8_folds(lab)}


def run_base():
    T27 = load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W; W.OUT = pathlib.Path(env.LOCAL) / "cv8_season"
    os.makedirs(CKB, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    for L, folds in layout_folds(W, lab).items():
        for k, fd in enumerate(folds):
            path = os.path.join(CKB, "%s_%d.csv" % (L, k))
            if os.path.exists(path): continue
            tm, vm = masks(lab, fd)
            if not vm.sum(): continue
            out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); out["fold"] = k
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "cv8_%s" % L, k)
            for s in SEEDS: out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
            save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = FC
            try: out["codex_R"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
            finally: W.TM.FEATURE_COLUMNS = save
            XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season; ps = []
            for seed in range(73, 81):
                idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
                m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed, ignore_pretraining_limits=True, inference_precision=torch.float32)
                m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
            out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
            out.to_csv(path, index=False); print("%s %d done (val %d, train %d, cold<=8 rows %d)" % (L, k, vm.sum(), tm.sum(), int((lab.in_temp[vm] <= 8).sum())), flush=True)

def run_gru():
    S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W; V = load("v3lf", "v3lf_field_linker_v3l_v1.py")
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(CKG, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
    S = V.sequences(lab, a, S6.USE, V.link_maps()["V3LF"])
    for L, folds in layout_folds(W, lab).items():
        for k, fd in enumerate(folds):
            path = os.path.join(CKG, "%s_%d.csv" % (L, k))
            if os.path.exists(path): continue
            tm, vm = masks(lab, fd)
            if not vm.sum(): continue
            out = pd.DataFrame({"row_id": lab.row_id[vm].values}); out["V3LF"] = np.mean([S7.fit_predict(S[tm], y[tm], wb[tm], S[vm], s, dev, "V3") for s in GSEEDS], 0)
            out.to_csv(path, index=False); print("%s/%d gru done | alone %.3f" % (L, k, rm(out.V3LF.values - y[vm])), flush=True)

def summarize():
    src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3os1"}
    exec(src[:src.index("JUDGE = [")], g)
    T, B, CT, A7 = g["T"], g["B"], g["CT"], g["A7"]; W = g.get("W")
    _X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
    _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
    O = pd.read_csv(os.path.join(env.LOCAL, "fl2_cv_FL.csv"), encoding="utf-8-sig"); O["dT_air"] = [TIN.at[r, 0] - TIN.at[p, 23] if p in TIN.index else np.nan for r, p in zip(O.rec, O.prev)]
    O = O[O.prev_is_labelled.astype(bool) & (O.dT_air.abs() <= 1.5) & O.anchor_st23.notna()].set_index("rec")
    _R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); _R = _R[_R.validator == "DIAG10"]
    _MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); _mt = np.array([_MT[(v, k)][r][h] for v, k, r, h in zip(_R.validator, _R.fold, _R.rec, _R.hour)])
    _R["NH"] = np.mean([(_R[f"A1_{s}"].values - .1 * _mt) / .9 for s in (1357, 2468, 3579)], axis=0)
    Eh = _R.groupby(["rec", "hour"]).NH.mean().reset_index().sort_values(["rec", "hour"]); Eh["cm"] = Eh.groupby("rec").NH.transform(lambda s: s.expanding().mean())
    CM = {(r, int(h)): c for r, h, c in zip(Eh.rec, Eh.hour, Eh.cm)}
    def anchors(M, q, trainrec, lo, hi, gate=None):
        """K8 (lo .8, hi 1.01, gate None) or cold-widening (lo .5, hi .8, soft p, row gate)."""
        out = np.zeros(len(q)); h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values
        sel = O[(O.p >= lo) & (O.p < hi)]; q0 = pd.Series(q[h == 0], index=key[h == 0])
        for rec in sel.index.intersection(q0.index):
            pv = sel.at[rec, "prev"]
            if pv not in trainrec.get(rec, ()): continue
            gap = np.clip(sel.at[rec, "anchor_st23"] - .36 - q0[rec], -2, 2) * (1.0 if gate is None else sel.at[rec, "p"])
            m = key == rec; out[m] = A7.BETA[h[m]] * gap * (1.0 if gate is None else gate[m])
        return out
    def tx3(M):
        rk = (M.farm + "_" + M.day.map("{:03d}".format)).values; cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(rk, M.hour.values)])
        return .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0)
    def gates(M, lo_g, lo_h):
        g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
        mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
    W2 = load("t27s", "tt27_chain_corrector_layoutL_v1.py").W; labw = W2.worlds()[0]; LF = layout_folds(W2, labw)
    allrec = sorted(set(labw.farm + "_" + labw.day.map("{:03d}".format)))
    def trainrec_for(folds, M):
        tr = {}
        for k, fd in enumerate(folds):
            tm, vm = masks(labw, fd); trs = set(labw.farm[tm] + "_" + labw.day[tm].map("{:03d}".format))
            for r in set(labw.farm[vm] + "_" + labw.day[vm].map("{:03d}".format)): tr[r] = trs
        return tr
    rng = np.random.default_rng(20270116); RES = {}
    for L in LAYS:
        M = pd.concat([pd.read_csv(os.path.join(CKB, f)) for f in sorted(os.listdir(CKB)) if f.startswith(L + "_")], ignore_index=True)
        Gd = pd.concat([pd.read_csv(os.path.join(CKG, f)) for f in sorted(os.listdir(CKG)) if f.startswith(L + "_")], ignore_index=True)
        M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
        y = M.sub_temp.values; v = M.V3LF.values; miss = np.isnan(v); TR = trainrec_for(LF[L], M); t3 = tx3(M)
        cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); gs = gates(M, 10, 12); go = gates(M, 8, 10)
        bk = (M.farm + "_" + (M.day // 5).astype(str)).values; P = {k: [] for k in ("BASE27", "C0", "C1", "C2", "C12")}
        for s in SEEDS:
            for fam in "AB":
                q = B.w40(M, "base_R_%d" % s, "codex_R", fam); k8 = anchors(M, q, TR, .8, 1.01)
                q1 = q + .10 * gs * (~miss) * (np.nan_to_num(v) - q); q0_ = q + .10 * go * (~miss) * (np.nan_to_num(v) - q)
                base = q + k8 + t3; c2add = anchors(M, q, TR, .5, .8, gate=cg)
                P["BASE27"].append(base); P["C0"].append(q0_ + anchors(M, q0_, TR, .8, 1.01) + t3); P["C1"].append(q1 + anchors(M, q1, TR, .8, 1.01) + t3)
                P["C2"].append(base + c2add); P["C12"].append(q1 + anchors(M, q1, TR, .8, 1.01) + t3 + anchors(M, q1, TR, .5, .8, gate=cg))
        cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179
        for c in ("C0", "C1", "C2", "C12"):
            ch = [rm(n - y) / rm(b - y) - 1 for n, b in zip(P[c], P["BASE27"])]
            dd = pd.Series(np.mean([(n - y) ** 2 - (b - y) ** 2 for n, b in zip(P[c], P["BASE27"])], 0)).groupby(bk).mean().values
            mb = dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1); pw, pb = float((mb >= 0).mean()), float((mb < 0).mean())
            sub = lambda m: np.mean([rm((n - y)[m]) / rm((b - y)[m]) - 1 for n, b in zip(P[c], P["BASE27"])]) * 100 if m.any() else np.nan
            RES[(L, c)] = (ch, pw, pb)
            print(f"{L:3s} {c:4s} vs BASE27 {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} P(better) {pb:.4f} | cold {sub(cold):+.2f}% warm {sub(~cold):+.2f}% p2 {sub(p2):+.2f}% (rows {len(y)}, cold {int(cold.sum())})", flush=True)
    def safe(L, c): ch, pw, pb = RES[(L, c)]; return not (np.mean(ch) > 0 and pb < .0125)
    def win(L, c): ch, pw, pb = RES[(L, c)]; return all(x < 0 for x in ch) and pw < .0125
    v1 = win("N", "C1") and safe("CK2", "C1"); v2 = win("CK2", "C2") and safe("N", "C2")
    print(f"OS1 C1 (GRU strict gate): {'ADOPT' if v1 else 'FAIL'} | C2 (cold anchor widening): {'ADOPT' if v2 else 'FAIL'}")

if __name__ == "__main__":
    {"base": run_base, "gru": run_gru, "sum": summarize}[MODE]()
