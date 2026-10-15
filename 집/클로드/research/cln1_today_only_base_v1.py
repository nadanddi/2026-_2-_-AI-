# -*- coding: utf-8 -*-
"""CLN1 (2026-10-16 집 클로드; 사용자 '진행해' — 5점 정리판 1단계: BASE 구성원을 현장 정보만으로).  사전 고정(커밋 후 실행).  온도.
문제(5점_부품별_적합성_점검_v1.md): BASE(W40G-S 비중 .4)의 특징에 `day`(= 기록 번호; 2차에선 달력 아님)와, 기록 번호 순으로 이어 붙인
  날 넘는 시차·이동 평균(lag·ewm·r24/72/168·dev6·d1·d3·hsince, 선형 부분 ph_* ewm 의 24·48·72h 등)이 있음 -> 대회 구조 의존.
BASE_TOD (오늘만):
  ct  = 원 ct − {day, *_lag*, *_ewm*, *_r24m/r24s/r72m/r168m, *_dev6, in_temp_d1·d3, *_hsince_*} + season(계절 지표, CODEX·PFN 과 같은 폴드 추정)
        + 기록 안 시차(같은 농장·같은 기록 안에서만 shift): in_temp 1·2·3h, out_temp 1h, rad_eff 1h, in_temp 1h·3h 차
  phc = 선형 물리 부분: ph_in_temp_None·ph_rad_None·ph_farm_id 유지, 날 넘는 ewm 은 같은 반감기의 기록 안 ewm(seg_*)으로:
        ph_in_temp_1 -> seg_in_temp_1, _3 -> seg_in_temp_2, _8 -> seg_in_temp_8, _24·_72 -> 뺌; ph_rad_2 -> seg_rad_2, _6 -> seg_rad_4;
        ph_heat_1·4 -> seg_heat_1·4; ph_out_temp_6 -> seg_out_temp_4, _48 -> 뺌.
  모형·가중·시드(7, 101)·폴드(Codex TK2 world, common.split_mask) = TT1 BASE 와 동일.  같은 실행에서 원 BASE(REF)도 다시 학습해 재현 확인.
-> local/cln1_ckpt/{배치}_{폴드}.csv (row_id, base_REF_7·101 재현, base_TOD_7·101).  판정은 별도 스크립트(CLN1J, 사전 고정).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln1_today_only_base_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"); sys.path.insert(0, TKDIR)
import world as W  # noqa: E402
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "cln1_ckpt"); SEEDS = (7, 101)
CROSS = lambda c: (c == "day" or "_lag" in c or "_ewm" in c or any(s in c for s in ("_r24m", "_r24s", "_r72m", "_r168m", "_dev6", "_hsince_")) or c in ("in_temp_d1", "in_temp_d3"))
PH_MAP = {"ph_in_temp_None": "ph_in_temp_None", "ph_rad_None": "ph_rad_None", "ph_farm_id": "ph_farm_id", "ph_in_temp_1": "seg_in_temp_1", "ph_in_temp_3": "seg_in_temp_2",
          "ph_in_temp_8": "seg_in_temp_8", "ph_rad_2": "seg_rad_2", "ph_rad_6": "seg_rad_4", "ph_heat_1": "seg_heat_1", "ph_heat_4": "seg_heat_4", "ph_out_temp_6": "seg_out_temp_4"}
WL = ["in_temp_wl1", "in_temp_wl2", "in_temp_wl3", "out_temp_wl1", "rad_eff_wl1", "in_temp_wd1", "in_temp_wd3"]
def main():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); assert (lab.row_id.values == pfn.row_id.values).all()
    lab = lab.copy(); g = lab.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"])
    for k in (1, 2, 3): lab[f"in_temp_wl{k}"] = g.in_temp.shift(k).reindex(lab.index)
    lab["out_temp_wl1"] = g.out_temp.shift(1).reindex(lab.index); lab["rad_eff_wl1"] = g.rad_eff.shift(1).reindex(lab.index)
    lab["in_temp_wd1"] = lab.in_temp - lab.in_temp_wl1; lab["in_temp_wd3"] = lab.in_temp - lab.in_temp_wl3
    ctT = [c for c in ct if not CROSS(c)] + ["season"] + WL; phcT = [PH_MAP[c] for c in phc if c in PH_MAP]
    print("ct", len(ct), "-> TOD", len(ctT), "| dropped:", [c for c in ct if CROSS(c)]); print("phc", phc, "-> TOD", phcT, flush=True)
    for name, folds in sets:
        if name not in ("DIAG10", "EXT10", "EXT12"): continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path): continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum(): continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "cln1_" + name, k)
            L = lab.copy(); L.loc[tm, "season"] = tr.season.values; L.loc[vm, "season"] = va.season.values
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True)
            for s in SEEDS:
                out["base_REF_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
                out["base_TOD_%d" % s] = W.base_predict(L[tm], L[vm], wb[tm], ctT, phcT, s)
            out.to_csv(path, index=False); y = lab.sub_temp.values[vm]
            print("%s/%d done | RMSE REF %.3f TOD %.3f" % (name, k, np.sqrt(np.mean((out.base_REF_7 - y) ** 2)), np.sqrt(np.mean((out.base_TOD_7 - y) ** 2))), flush=True)
if __name__ == "__main__":
    main()
