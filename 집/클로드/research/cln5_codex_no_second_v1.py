# -*- coding: utf-8 -*-
"""CLN5 (2026-10-16 집 클로드; 5점 정리판 — CODEX·TabPFN 특징 'second' = (기록 번호 >= 179) 는 대회 자료 1·2차 구분(번호 구조) -> 빼는 비용 측정).
사전 고정(커밋 후 실행).  온도.  CODEX 만(CPU); TabPFN 은 결과 보고 판단.
CODEX 는 하루 안 특징(h0·mean·std·diff·reset; 0시에 새로 시작)뿐이라 평가형 세계와 무관(CLN4 점검) -> TT1 과 같은 W world·폴드(common.split_mask)·season_fold·시드 726.
같은 실행에서 두 판 학습: codex_S(FEATURE_COLUMNS day -> season, 현행 재현) / codex_NS(같고 'second' 뺌).  배치 DIAG10·EXT10·EXT12.
-> local/cln5_ckpt/{배치}_{폴드}.csv (row_id, codex_S, codex_NS).  판정은 CLN5J(사전 고정).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln5_codex_no_second_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"); sys.path.insert(0, TKDIR)
import world as W  # noqa: E402
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "cln5_ckpt"); rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
def main():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); FC = list(W.FEATURE_COLUMNS); assert "second" in FC
    FS = [c if c != "day" else "season" for c in FC]; FNS = [c for c in FS if c != "second"]
    for name, folds in sets:
        if name not in ("DIAG10", "EXT10", "EXT12"): continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path): continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum(): continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "cln5_" + name, k)
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True); save = W.TM.FEATURE_COLUMNS
            for tag, cols in (("S", FS), ("NS", FNS)):
                W.TM.FEATURE_COLUMNS = cols
                try: out["codex_" + tag] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
                finally: W.TM.FEATURE_COLUMNS = save
            out.to_csv(path, index=False); y = lab.sub_temp.values[vm]
            print("%s/%d done | CODEX RMSE S %.4f NS %.4f" % (name, k, rm(out.codex_S - y), rm(out.codex_NS - y)), flush=True)
if __name__ == "__main__":
    main()
