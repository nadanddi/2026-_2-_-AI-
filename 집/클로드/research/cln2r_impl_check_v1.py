# -*- coding: utf-8 -*-
"""CLN2R (2026-10-16 집 클로드; CLN2·CLN3(7.033·7.034): 추운 행 손해가 '재연결' 때문인지 '학습 구현 차이'(RS1 TF1.fit_base vs 기준 W.base_predict) 때문인지 가르기).
서술 점검, 실행 전 커밋.  CLN2 와 같은 장치·폴드에서 링커 없이(RS1 Raw 기본 = 앞 번호로 이어 = 원래 번호 순 세계 그대로) BASE 학습 -> CLN1 재학습 REF 와 비교.
배치·폴드: EXT10:0, EXT12:0 (추운 행 많음).  출력: 전체·실내 <= 8도·<= 6도 RMSE (REF_cln / NUM_fitbase / RS(CLN2) / TOD).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cln2r_impl_check_v1.py
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("cln2", os.path.join(HERE, "cln2_fl_restitch_base_v1.py")); C2 = importlib.util.module_from_spec(s); s.loader.exec_module(C2)
import json
R1, EVP, TF1, W, common = C2.R1, C2.EVP, C2.TF1, C2.W, C2.common
C = json.load(open(os.path.join(EVP.AUD, "cols.json"))); ct, phc = C["ct"], C["phc"]
L0 = pd.read_pickle(os.path.join(EVP.AUD, "lab_feats.pkl")); wmap = dict(zip(L0.row_id, L0.w))
lab, pfn, _, _, wb, wp, wv, sets, z = W.worlds()
class RawNum(R1.Raw):
    def __init__(self):
        super().__init__(); self.link = {}            # step = previous number everywhere (original number-order world)
raw = RawNum(); rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
C1 = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "cln1_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "cln1_ckpt"))]).drop_duplicates("row_id")
for L, k in (("EXT10", 0), ("EXT12", 0)):
    fd = dict(sets)[L][k]; tm, vm = common.split_mask(lab, fd)
    labrec = set(zip(lab.farm, lab.day.astype(int))); val = set(zip(lab.farm[vm], lab.day[vm].astype(int))); trn = set(zip(lab.farm[tm], lab.day[tm].astype(int))); dele = labrec - val - trn
    vis_tr = lambda r: (lambda x: x in raw.train and x not in val and x not in dele)
    TR = R1.build_rows(raw, sorted(trn), val | dele, True, vis_tr, ct, phc, [])
    def vis_va(r):
        def v(x):
            if x in dele: return False
            if x in val and (x[0] != r[0] or x[1] >= r[1]): return False
            if x in raw.test and x[0] != r[0]: return False
            return True
        return v
    VA = R1.build_rows(raw, sorted(val), dele, False, vis_va, ct, phc, [])
    w = TR.row_id.map(wmap).values.astype(float)
    O = VA[["row_id", "sub_temp", "in_temp"]].copy(); O["NUM"] = TF1.fit_base(TR, ct, phc, w, 7)(VA)
    O = O.merge(C1[["row_id", "base_REF_7", "base_TOD_7"]], on="row_id", how="left").merge(pd.read_pickle(os.path.join(env.LOCAL, "cln2_ckpt", "%s_%d.pkl" % (L, k)))[["row_id", "base_RS_7"]], on="row_id", how="left")
    y = O.sub_temp.values
    for nm, m in (("all", np.ones(len(O), bool)), ("<=8C", (O.in_temp <= 8).values), ("<=6C", (O.in_temp <= 6).values)):
        print(f"{L}/{k} {nm:5s} n {int(m.sum()):5d} | REF(W.base_predict) {rm((O.base_REF_7 - y)[m]):.3f} | NUM(TF1.fit_base, number order) {rm((O.NUM - y)[m]):.3f} | RS(FL relink) {rm((O.base_RS_7 - y)[m]):.3f} | TOD {rm((O.base_TOD_7 - y)[m]):.3f}", flush=True)
    print("  corr REF vs NUM %.4f" % np.corrcoef(O.base_REF_7, O.NUM)[0, 1])
