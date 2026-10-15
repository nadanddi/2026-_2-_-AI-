# -*- coding: utf-8 -*-
"""RVT (2026-10-15 집 클로드; 사용자: '온도 쪽에도 이번에 EC 모델 발견한 것처럼 진행해봐').  서술(사후), 실행 전 커밋.  온도.
RV2(EC)처럼 온도 후보마다 '어느 상황에서 확실히 좋아지나' 를 같은 틀로 분해 → 모든 판에서 개선되는 칸 = 추출 후보(다음 정식 판정용).
틀: TV3 v23(W40G-S → 사슬 C2 → 앵커 v7 + TNA8), 판 DIAG10·EL1·F·G·H·I(+ EXT10·EXT12 서술), 셀 4(BASE 시드 x PFN 계열).
상황: 링크(26번 규칙 MASK 거울 링커 p0 ≥ .5) 있음/없음, 추운 행(in_temp ≤ 8)/따뜻한 행, 시각 0-7 / 8-15 / 16-23, 2차.
후보(기준):
  V3L(V3)            = v3l_ckpt V3L 10% vs 저장 V3 10% (날 문턱 g·h 같음)
  V3T(V3L)           = 진짜 전날 창(현장 조건) vs V3L
  V3L30link(V3L)     = 링크 있는 기록만 V3L 30% (나머지 10%) — 사후 아이디어
  TRUEanch(CUR)      = 앵커·사슬표를 CLEAN 진짜 전날로(C1, 현장 조건) vs 링커
  ANCHlink_only(CUR) = 링커 앵커를 MASK 거울 링크가 있는 기록에만 남김 — 앵커가 어디서 이득인지
보고: 상황별 셀 평균 상대 RMSE %, '6판 모두 개선' 표시, 상황별 기록 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u rvt_temp_success_extract_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "c1_true_yesterday_field_condition_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "c1_true_yesterday_field_condition_v1.py")
exec(src[:src.index('print("[TEMP]", flush=True)')])   # tv3 harness + TRUE, CTT, LT
__file__ = os.path.join(HERE, "rvt_temp_success_extract_v1.py")
CKL = os.path.join(env.LOCAL, "v3l_ckpt")
Lm = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); MLINK = {r for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .5}
def ldv(L):
    fs = [f for f in os.listdir(CKL) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(CKL, f)) for f in fs]).drop_duplicates("row_id")
rows = []; JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I"]
for L in JUDGE + ["EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); MT = T.layout_frame(L, CTT); y = M.sub_temp.values
    X = M[["row_id"]].merge(ldv(L)[["row_id", "V3L", "V3T"]], on="row_id", how="left"); v3 = v3_for(L, M)
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
    rec = (M.farm + "_" + M.day.map("{:03d}".format)).values; lk = np.isin(rec, list(MLINK)); cold = (M.in_temp <= 8).values; hr = M.hour.values; p2 = M.day.values >= 179
    LVl = LV[LV.rec.isin(MLINK)]
    P = {k: [] for k in ("V3", "V3L", "V3T", "V3L30link", "TRUEanch", "CURanch", "ANCHlink")}
    def mix(a, v, w): miss = np.isnan(v); return a + w * g_ * h_ * (~miss) * (np.nan_to_num(v) - a)
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam); aT = B.w40(MT, b, cx, fam)
            P["V3"].append(anch(M, C2.corrected(M, mix(a, v3, .1), D1, bi, fam, 7), LV))
            P["V3L"].append(anch(M, C2.corrected(M, mix(a, X.V3L.values, .1), D1, bi, fam, 7), LV))
            P["V3T"].append(anch(M, C2.corrected(M, mix(a, X.V3T.values, .1), D1, bi, fam, 7), LV))
            P["V3L30link"].append(anch(M, C2.corrected(M, mix(a, X.V3L.values, np.where(lk, .3, .1)), D1, bi, fam, 7), LV))
            P["CURanch"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV))
            P["TRUEanch"].append(anch(MT, C2.corrected(MT, aT, D1T, bi, fam, 7), LT))
            P["ANCHlink"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LVl))
    segs = {"all": np.ones(len(y), bool), "linked": lk, "unlinked": ~lk, "cold": cold, "warm": ~cold, "h0-7": hr <= 7, "h8-15": (hr >= 8) & (hr <= 15), "h16-23": hr >= 16, "p2": p2}
    for name, cand, base in (("V3L vs V3", "V3L", "V3"), ("V3T vs V3L", "V3T", "V3L"), ("V3L30link vs V3L", "V3L30link", "V3L"),
                             ("TRUEanch vs CURanch", "TRUEanch", "CURanch"), ("ANCHlink-only vs CURanch", "ANCHlink", "CURanch")):
        for sn, m in segs.items():
            if not m.any(): continue
            ch = np.mean([rm((c - y)[m]) / rm((b_ - y)[m]) - 1 for c, b_ in zip(P[cand], P[base])])
            rows.append(dict(model=name, val=L, seg=sn, pct=100 * ch, n_rec=int(len(set(rec[m])))))
    print("done", L, flush=True)
R = pd.DataFrame(rows); pd.set_option("display.width", 260)
Pv = R.pivot_table(index=["model", "seg"], columns="val", values="pct").round(2); Pv["all6_better"] = (Pv[JUDGE] < 0).all(axis=1)
print(Pv[JUDGE + ["EXT10", "EXT12", "all6_better"]].to_string())
print("\nrecords per seg (DIAG10):", R[(R.val == "DIAG10") & (R.model == "V3L vs V3")].set_index("seg").n_rec.to_dict())
Pv.to_csv(os.path.join(HERE, "rvt_extract_table_v1.csv"))
