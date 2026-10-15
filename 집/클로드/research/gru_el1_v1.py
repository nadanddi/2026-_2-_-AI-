# -*- coding: utf-8 -*-
"""GRU_EL1 (2026-10-15 집 클로드; CB1-T 준비).  OS1 v2 의 V3LF(FL 학습표 p >= .5 창, 시드 3·4) 를 EL1 폴드(common.split_mask)에 학습 → os1_gru_ckpt/EL1_k.csv.  판정 없음(예측 생성만)."""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
O1 = load("os1v2", "os1_temp_regime_split_v2.py"); S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W; V = load("v3lf", "v3lf_field_linker_v3l_v1.py")
import env_extra_gpu  # noqa: F401
import torch, common
dev = "cuda" if torch.cuda.is_available() else "cpu"
lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
S = V.sequences(lab, a, S6.USE, V.link_maps()["V3LF"])
for k, fd in enumerate(dict(sets)["EL1"]):
    path = os.path.join(O1.CKG, "EL1_%d.csv" % k)
    if os.path.exists(path): continue
    tm, vm = common.split_mask(lab, fd)
    if not vm.sum(): continue
    out = pd.DataFrame({"row_id": lab.row_id[vm].values}); out["V3LF"] = np.mean([S7.fit_predict(S[tm], y[tm], wb[tm], S[vm], s, dev, "V3") for s in O1.GSEEDS], 0)
    out.to_csv(path, index=False); print("EL1/%d gru done" % k, flush=True)
