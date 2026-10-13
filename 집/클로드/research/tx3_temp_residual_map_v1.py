# -*- coding: utf-8 -*-
"""TX3 (2026-10-13 집 클로드; 사용자: '온도 쪽 더 파봐'). 서술: 13회차 구성(+TN8) 온도 잔차가 어디 몰려 있나.
배치 DIAG10(2차)·EXT10·EXT12.  기록 하루 수준 오차(평균 잔차)와 하루 안 모양 오차로 나누고, 상위 기록의 성질(앵커 여부, 농장, 추움, EC 예측, 0시 오차)을 표로.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import ST, FARM, NUM, N  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
R = A7.A2.A1.frames()
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
TN8 = {r: pv for r, pv, p0, lb in zip(PM.index, PM.prev0, PM.p0, PM.lab0) if r not in v7set and bool(lb) and isinstance(pv, str) and p0 >= .8}
add = pd.DataFrame(dict(rec=list(TN8), prev=list(TN8.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TN8.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True); anch = set(LC.rec)
E = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); E = E[(E.tag == "CUR") & (E.validator == "DIAG10")]; ecsg = E.groupby(E.row_id.str[:7]).sg.mean()
pd.set_option("display.width", 230)
for L, sub in (("DIAG10", "p2"), ("EXT10", "all"), ("EXT12", "all")):
    M, cells = R[L]; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; kb, Ab = A7.build(M, LC)
    q = np.mean([A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells], axis=0); y = M.sub_temp.values
    D = pd.DataFrame(dict(rec=key, h=M.hour.values, e=q - y, inT=M.in_temp.values, day=M.day.values))
    if sub == "p2": D = D[D.day >= 179]
    g = D.groupby("rec").agg(off=("e", "mean"), e0=("e", "first"), sse=("e", lambda s: float(np.sum(s ** 2))), inT=("inT", "mean"))
    g["shape_sse"] = D.groupby("rec").e.apply(lambda s: float(np.sum((s - s.mean()) ** 2))); g["off_sse"] = g.sse - g.shape_sse
    g["farm"] = g.index.str[:3]; g["anchored"] = g.index.isin(anch); g["ecsg"] = ecsg.reindex(g.index).values; g["cold"] = g.inT < 10
    tot = g.sse.sum()
    print(f"\n== {L} ({sub}): records {len(g)} | total SSE {tot:.1f} | day-offset share {g.off_sse.sum() / tot:.2f} | F47 share {g[g.farm == 'F47'].sse.sum() / tot:.2f} | anchored share {g[g.anchored].sse.sum() / tot:.2f} (n {int(g.anchored.sum())}) | cold(<10C mean) share {g[g.cold].sse.sum() / tot:.2f} | top10% share {g.sse.nlargest(max(1, len(g) // 10)).sum() / tot:.2f}")
    print("  offset sign among top-10 SSE records: over-pred", int((g.nlargest(10, "sse").off > 0).sum()), "/ 10")
    print(g.nlargest(10, "sse")[["sse", "off", "e0", "inT", "farm", "anchored", "ecsg"]].round(2).to_string())
