# -*- coding: utf-8 -*-
"""TX1 (2026-10-13 집 클로드; 사용자: '온도쪽을 더 보자'). 서술.
HX9 근권(6.797): 고EC 동은 배지가 입력 모델보다 +.25℃ 따뜻, 밤 난방 1/4.  → 온도 최종(13회차 구성: TC1c 셀 + v7 앵커, DIAG10 계열)의 기록 잔차가
  (a) 실제 EC 수준(학습 정답), (b) EC 모델 예측 sg(DIAG10 OOF, 대상 정답 미사용 — 평가에서도 계산 가능), (c) 밤 난방 수준과 어떻게 엮이는지.  시각대별(0-7, 8-15, 16-23)도.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
R = A7.A2.A1.frames()
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
E = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); E = E[(E.tag == "CUR") & (E.validator == "DIAG10")]
ecs = E.groupby(E.row_id.str[:7]).agg(ec_sg=("sg", "mean"), ec_y=("sub_ec", "mean"))
_X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
heat_n = _X[(_X.h <= 5) | (_X.h >= 20)].groupby("rec").act_heating.mean().rename("heat_n")
out = []
for L in ("DIAG10", "EXT10", "EXT12", "EL1"):
    M, cells = R[L]; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; kb, Ab = A7.build(M, L7f)
    q = np.mean([A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells], axis=0); res = M.sub_temp.values - q
    D = pd.DataFrame(dict(rec=key, h=M.hour.values, res=res, p2=M.day.values >= 179, cold=M.in_temp.values <= 8))
    for blk, (a, b) in {"all": (0, 23), "0-7": (0, 7), "8-15": (8, 15), "16-23": (16, 23)}.items():
        g = D[(D.h >= a) & (D.h <= b)].groupby("rec").agg(res=("res", "mean"), p2=("p2", "first")).join(ecs).join(heat_n)
        g["hiEC"] = g.ec_y >= 1.0; g["sgHi"] = g.ec_sg >= 1.0
        out.append(dict(layout=L, hours=blk, n=len(g), res_hiEC=g[g.hiEC].res.mean(), res_norm=g[~g.hiEC].res.mean(), n_hi=int(g.hiEC.sum()),
                        res_sgHi=g[g.sgHi].res.mean(), res_sgLo=g[~g.sgHi].res.mean(), r_res_ecy=g[["res", "ec_y"]].corr().iloc[0, 1], r_res_ecsg=g[["res", "ec_sg"]].corr().iloc[0, 1],
                        r_res_heatn=g[["res", "heat_n"]].corr().iloc[0, 1], r_res_ecsg_p2=g[g.p2][["res", "ec_sg"]].corr().iloc[0, 1]))
pd.set_option("display.width", 220); print(pd.DataFrame(out).round(3).to_string(index=False))
