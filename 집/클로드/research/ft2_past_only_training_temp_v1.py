# -*- coding: utf-8 -*-
"""FT2 (2026-10-16 집 클로드; 사용자 'c2 진행' — 5점 근거 C2 온도판).  서술 측정(채택 판정 아님), 실행 전 커밋.  온도.
질문: 현장 모델은 대상 날짜보다 '앞선' 날의 정답만 학습할 수 있다.  온도 기본 단계(W40G-S 셀: .4 BASE + (.2 + .4(1-g)) CODEX + .4 g PFN, g = clip((실내 - 8)/2))가
  과거만 학습해도 유지되는가, 날짜(계절) 이득이 '미래 이웃 정답 빌리기'인가.  복원 단계(앵커·사슬·TX3)는 빼고 기본 모델만 잼.
대상 날·날짜(CAL)·잠금: FT1(6.910, EC) 과 같음(ft1.setup: 정답 있는 모든 날을 dc4.season_index 로 한 번 추정한 날짜를 '알려진 날짜'로 고정; 2차 전부 + 1차 CAL >= 60 의 6번째마다).
학습 배치: PAST = CAL < 대상 CAL (두 농장, 잠금 ±1 제외) / ALL = FT1 과 같은 숨김(같은 농장 ±1, 다른 농장 −3..+3, 잠금 ±1).
구성원(OS1 v2 run_base 와 같은 함수): BASE = W.base_predict 시드 1616·1717, CODEX = W.TM.codex_fit_predict(시드 726, season = CAL), PFN = TabPFN v2 문맥 2000 x 8 (73~80; A = 앞 4, B = 뒤 4), season = CAL.
보고(sum): 셀·구성원 RMSE (전체 / 2차 / 1차 / 실내 <= 8 / CAL 구간), PAST vs ALL 변화, 학습 날 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ft2_past_only_training_temp_v1.py {list|smoke|cv|sum}
  smoke : 첫 대상 하루만 BASE·CODEX (PFN 없음, CPU) — 코드 점검용, 저장 안 함
"""
import env  # noqa: F401
import importlib.util, os, sys, gc, glob
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "list"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "list"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CK = os.path.join(env.LOCAL, "ft2_ckpt"); SEEDS = (1616, 1717); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
OTHER = {"F13": "F47", "F47": "F13"}
def setup():
    FT1 = load("ft1", "ft1_past_only_training_ec_v1.py"); _, lockd, CAL, tg, _, _, _ = FT1.setup()
    T27 = load("t27", "tt27_chain_corrector_layoutL_v1.py"); W = T27.W
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    lab = lab.reset_index(drop=True); pfn = pfn.reset_index(drop=True); assert len(lab) == len(pfn) and (lab.row_id.values == pfn.row_id.values).all()
    key = list(zip(lab.farm, lab.day.astype(int))); cal = np.array([CAL.get(k, np.nan) for k in key])
    tg = [t for t in tg if t in set(key)]
    return W, lab, pfn, ct, phc, wb, wp, lockd, CAL, cal, key, tg
def masks(key, cal, lockd, CAL, t):
    f, d = t; c = CAL[t]; lk = {(ff, dd + j) for ff, dd in lockd for j in (-1, 0, 1)}
    past = np.array([np.isfinite(cc) and cc < c and k not in lk for k, cc in zip(key, cal)])
    forb = {(f, d + j) for j in (-1, 0, 1)} | {(OTHER[f], d + j) for j in range(-3, 4)} | lk
    allm = np.array([k not in forb for k in key]); va = np.array([k == t for k in key])
    return past, allm, va
def members(W, lab, pfn, ct, phc, wb, wp, cal, tm, vm, with_pfn=True):
    out = {}
    for s in SEEDS: out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    tr = pfn[tm].assign(season=cal[tm]).reset_index(drop=True); va = pfn[vm].assign(season=cal[vm]).reset_index(drop=True)
    save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = FC
    try: out["codex_R"] = W.TM.codex_fit_predict(tr, va, wp[tm], 726)
    finally: W.TM.FEATURE_COLUMNS = save
    if with_pfn:
        import env_extra_gpu  # noqa: F401
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        ixday = list(W.FEATURE_COLUMNS).index("day"); X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy()
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = cal[tm]; VS[:, ixday] = cal[vm]; ps = []
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for seed in range(73, 81):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=dev, n_estimators=4, random_state=seed, ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect()
            if dev == "cuda": torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
    return out
def run(mode):
    W, lab, pfn, ct, phc, wb, wp, lockd, CAL, cal, key, tg = setup(); print("targets", len(tg), "(pass-2", sum(d >= 179 for _, d in tg), ")", flush=True)
    if mode == "list":
        for t in tg:
            past, allm, va = masks(key, cal, lockd, CAL, t); print(t, "CAL %.1f" % CAL[t], "train days PAST", len(set(k for k, m in zip(key, past) if m)), "ALL", len(set(k for k, m in zip(key, allm) if m)), "rows", int(va.sum()))
        return
    if mode == "smoke":
        t = tg[len(tg) // 2]; past, allm, va = masks(key, cal, lockd, CAL, t); o = members(W, lab, pfn, ct, phc, wb, wp, cal, past, va, with_pfn=False)
        y = lab.sub_temp.values[va]; print("smoke", t, {k: round(rm(v - y), 3) for k, v in o.items()}); return
    os.makedirs(CK, exist_ok=True)
    for t in tg:
        path = os.path.join(CK, "%s_%03d.csv" % t)
        if os.path.exists(path): continue
        past, allm, va = masks(key, cal, lockd, CAL, t)
        G = lab.loc[va, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); G["cal"] = CAL[t]
        for reg, m in (("PAST", past), ("ALL", allm)):
            G[f"ntrain_{reg}"] = len(set(k for k, mm in zip(key, m) if mm))
            for c, v in members(W, lab, pfn, ct, phc, wb, wp, cal, m, va).items(): G[f"{reg}_{c}"] = v
        G.to_csv(path, index=False); print("%s_%03d done" % t, flush=True)
def w40(D, reg, s, fam):
    g = np.where(D.in_temp.isna(), 1, np.clip((D.in_temp - 8) / 2, 0, 1))
    return (.4 * D[f"{reg}_base_R_{s}"] + (.2 + .4 * (1 - g)) * D[f"{reg}_codex_R"] + .4 * g * D[f"{reg}_pfn_{fam}"]).values
def summarize():
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CK, "*.csv"))], ignore_index=True); y = D.sub_temp.values
    D["p2"] = D.day >= 179
    sc = {"all": np.ones(len(D), bool), "p2": D.p2.values, "p1": ~D.p2.values, "in<=8": (D.in_temp <= 8).values, "cal<60": (D.cal < 60).values,
          "cal60_90": ((D.cal >= 60) & (D.cal < 90)).values, "cal90_120": ((D.cal >= 90) & (D.cal < 120)).values, "cal120+": (D.cal >= 120).values}
    print("target days", D.groupby(["farm", "day"]).ngroups, "rows", len(D))
    rows = []
    for reg in ("PAST", "ALL"):
        cells = [w40(D, reg, s, fam) for s in SEEDS for fam in "AB"]
        for nm, preds in (("cell", cells), ("BASE", [D[f"{reg}_base_R_{s}"].values for s in SEEDS]), ("CODEX", [D[f"{reg}_codex_R"].values]), ("PFN", [D[f"{reg}_pfn_{f}"].values for f in "AB"])):
            r = dict(arm=f"{reg}_{nm}")
            for k, m in sc.items(): r[k] = np.mean([rm((p - y)[m]) for p in preds]) if m.any() else np.nan
            rows.append(r)
    R = pd.DataFrame(rows).set_index("arm"); pd.set_option("display.width", 220); print(R.round(4).to_string())
    for nm in ("cell", "BASE", "CODEX", "PFN"): print(f"PAST vs ALL {nm} (%):", (100 * (R.loc[f"PAST_{nm}"] / R.loc[f"ALL_{nm}"] - 1)).round(1).to_dict())
    print("train days PAST median", float(D.groupby(["farm", "day"]).ntrain_PAST.first().median()), "ALL median", float(D.groupby(["farm", "day"]).ntrain_ALL.first().median()))
if __name__ == "__main__":
    summarize() if MODE == "sum" else run(MODE)
