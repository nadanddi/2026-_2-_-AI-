# -*- coding: utf-8 -*-
"""CLN6 (2026-10-16 집 클로드; 5점 정리판 마지막 — TabPFN 특징 'second'(기록 번호 >= 179) 제거.  CLN5: CODEX 에선 비용 ≈ 0).
사전 고정(커밋 후 실행).  온도.  이 컴퓨터 GPU(RTX 4050).
OS1 v2 run_base 의 TabPFN 조리법 그대로(W world, common.split_mask 폴드, season_fold 의 계절을 day 자리에, 문맥 2000행 wp 가중 추출, 시드 73~80, A = 앞 4, B = 뒤 4, n_estimators 4).
같은 실행에서 두 판: pfn_S(현행 특징, second 포함) / pfn_NS(second 열 뺌).  배치 DIAG10·EXT10·EXT12.
-> local/cln6_ckpt/{배치}_{폴드}.csv (row_id, pfn_S_A, pfn_S_B, pfn_NS_A, pfn_NS_B).  판정은 CLN6J(사전 고정).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln6_pfn_no_second_v1.py
"""
import os, sys, gc
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"); sys.path.insert(0, TKDIR)
import world as W  # noqa: E402
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "cln6_ckpt"); rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
def main():
    os.makedirs(CK, exist_ok=True); assert torch.cuda.is_available()
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wp = np.asarray(wp, float)
    FC = list(W.FEATURE_COLUMNS); ixday = FC.index("day"); ixsec = FC.index("second"); keep = [i for i in range(len(FC)) if i != ixsec]
    X = pfn[FC].to_numpy(np.float32); y = pfn.sub_temp.to_numpy()
    for name, folds in sets:
        if name not in ("DIAG10", "EXT10", "EXT12"): continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path): continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum(): continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "cln6_" + name, k)
            XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season.values; VS[:, ixday] = va.season.values
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True)
            for tag, cols in (("S", list(range(len(FC)))), ("NS", keep)):
                ps = []
                for seed in range(73, 81):
                    idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
                    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed, ignore_pretraining_limits=True, inference_precision=torch.float32)
                    m.fit(XS[idx][:, cols], y[tm][idx]); ps.append(m.predict(VS[:, cols])); del m; gc.collect(); torch.cuda.empty_cache()
                out["pfn_%s_A" % tag] = np.mean(ps[:4], 0); out["pfn_%s_B" % tag] = np.mean(ps[4:], 0)
            out.to_csv(path, index=False); yy = lab.sub_temp.values[vm]
            print("%s/%d done | PFN RMSE S %.4f NS %.4f" % (name, k, rm(out.pfn_S_A - yy), rm(out.pfn_NS_A - yy)), flush=True)
if __name__ == "__main__":
    main()
