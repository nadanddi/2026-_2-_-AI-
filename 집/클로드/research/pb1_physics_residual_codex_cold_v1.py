# -*- coding: utf-8 -*-
"""PB1 (2026-10-15 집 클로드; 5점모델_설계_v1.md 1층 — 물리 기준선 차이 학습, 추운 구간).  사전 고정(커밋 후 실행).  온도.
확인: BASE 의 주 구성원 res(.65) 는 이미 '물리 선형식(ph_* 지연 특징) + LightGBM 차이' 구조. 남은 새 요소 = CODEX 도 차이를 학습하게 하고 추운 행에만 쓰기.
  근거 H-DT1(6b.26): 나무·커널 목표를 sub − ph_in_temp_3 로 → DIAG10 +1.45% 이나 추운 EXT −0.7~−3.2% (구간 효과).
CODEX_P = 같은 CODEX(시드 726, 같은 특징·가중·계절 폴드) 를 목표 y − ph_in_temp_3 로 학습 → 예측 + ph_in_temp_3.  배치 = OS1 v2 의 N·CK2(같은 폴드).
기준 B = W40G-S 셀(os1_base_ckpt; BASE 1616/1717 x PFN A/B) → 27번 꼴 FL K8 앵커(OS1 과 같은 함수) + TX3.
후보 PB = 같은 계산에서 CODEX 를  cx' = g_c·CODEX_P + (1 − g_c)·CODEX,  g_c = clip((10 − in_temp)/2, 0, 1)(결측 0)  로 바꿈(새 상수 없음; TX3·C2 와 같은 추운 게이트).
판정(단일 안 α .025): CK2 4셀 모두 개선 & CK2 블록 P(악화) < .025 & N 안전(셀 평균 악화 & P(나음) < .025 이면 FAIL).  서술: 추운 행·2차, CODEX_P 단독 RMSE.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pb1_physics_residual_codex_cold_v1.py {fit|sum}
"""
import os, sys, importlib.util, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "sum"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
O1 = load("os1v2", "os1_temp_regime_split_v2.py"); CKP = os.path.join(env.LOCAL, "pb1_ckpt"); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))

def fit():
    T27 = O1.load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W; W.OUT = pathlib.Path(env.LOCAL) / "os1_season"; os.makedirs(CKP, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]; ph = lab.ph_in_temp_3.values; assert np.isfinite(ph).all()
    for L, folds in O1.layout_folds(W, lab).items():
        for k, fd in enumerate(folds):
            path = os.path.join(CKP, "%s_%d.csv" % (L, k))
            if os.path.exists(path): continue
            tm, vm = O1.masks(lab, fd)
            if not vm.sum(): continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "os1_%s" % L, k)
            tr = tr.reset_index(drop=True).copy(); va = va.reset_index(drop=True).copy(); tr["sub_temp"] = tr.sub_temp.values - ph[tm]
            save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = FC
            try: p = W.TM.codex_fit_predict(tr, va, wp[tm], 726)
            finally: W.TM.FEATURE_COLUMNS = save
            out = pd.DataFrame({"row_id": lab.row_id[vm].values, "codex_P": np.asarray(p) + ph[vm]}); out.to_csv(path, index=False)
            print("%s %d done | CODEX_P alone %.3f" % (L, k, rm(out.codex_P.values - lab.sub_temp[vm].values)), flush=True)

def summarize():
    src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
    g = {"__file__": os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), "__name__": "kf3pb"}; exec(src[:src.index("rng = np.random.default_rng(20270118)")], g)
    B, BETA, LINK, IDX, ST, IN_, RD, CM, lab, LF = g["B"], g["BETA"], g["LINK"], g["IDX"], g["ST"], g["IN_"], g["RD"], g["CM"], g["lab"], g["LF"]
    FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
    rng = np.random.default_rng(20270121); ok = True; res = {}
    for L in O1.LAYS:
        M = pd.concat([pd.read_csv(os.path.join(O1.CKB, f)) for f in sorted(os.listdir(O1.CKB)) if f.startswith(L + "_")], ignore_index=True)
        P = pd.concat([pd.read_csv(os.path.join(CKP, f)) for f in sorted(os.listdir(CKP)) if f.startswith(L + "_")], ignore_index=True)
        M = M.merge(P, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
        y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; pos = {r: np.where(key == r)[0] for r in set(key)}
        cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
        tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg
        TRS = {}
        for k, fd in enumerate(LF[L]):
            tm, vm = O1.masks(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
            for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
        K8 = {r: float(ST[IDX[FLK[r][0]], 23]) for r in pos if r in FLK and FLK[r][1] >= .8 and FLK[r][0] in TRS[r] and np.isfinite(ST[IDX[FLK[r][0]], 23])
              and abs(IN_[IDX[r], 0, 0] - IN_[IDX[FLK[r][0]], 23, 0]) <= 1.5}
        M2 = M.copy(); M2["codex_R"] = cg * M.codex_P.values + (1 - cg) * M.codex_R.values
        def full(MM, s, fam):
            c = B.w40(MM, "base_R_%d" % s, "codex_R", fam); b = c + tx3
            for r, A in K8.items():
                ix = pos[r]; q0 = c[ix][h[ix] == 0]
                if len(q0): b[ix] += BETA[h[ix]] * np.clip(A - .36 - q0[0], -2, 2)
            return b
        PB = [full(M, s, fam) for s in O1.SEEDS for fam in "AB"]; PP = [full(M2, s, fam) for s in O1.SEEDS for fam in "AB"]
        ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(PP, PB)]; bk = (M.farm + "_" + (M.day // 5).astype(str)).values
        dd = pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(PP, PB)], 0)).groupby(bk).mean().values
        mb = dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1); pw, pb = float((mb >= 0).mean()), float((mb < 0).mean()); res[L] = (ch, pw, pb)
        cold = (M.in_temp <= 8).values; p2 = M.day.values >= 179
        sub = lambda m: 100 * np.mean([rm((p - y)[m]) / rm((b - y)[m]) - 1 for p, b in zip(PP, PB)]) if m.any() else np.nan
        print(f"{L:3s} PB vs B {' '.join('%+.3f' % (100 * x) for x in ch)} | P(worse) {pw:.4f} P(better) {pb:.4f} | cold {sub(cold):+.2f}% p2 {sub(p2):+.2f}% | "
              f"rows gated {int((cg > 0).sum())} | CODEX alone {rm(M.codex_R - y):.3f} CODEX_P alone {rm(M.codex_P - y):.3f} (cold rows: {rm((M.codex_R - y)[cold]):.3f} / {rm((M.codex_P - y)[cold]):.3f})", flush=True)
    ch, pw, pb = res["CK2"]; c1 = all(x < 0 for x in ch) and pw < .025; chN, pwN, pbN = res["N"]; c2 = not (np.mean(chN) > 0 and pbN < .025)
    print("PB1:", "ADOPT" if c1 and c2 else "FAIL")

if __name__ == "__main__":
    {"fit": fit, "sum": summarize}[MODE]()
