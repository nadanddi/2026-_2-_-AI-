# -*- coding: utf-8 -*-
"""BMX1 (2026-10-16 집 클로드; 사용자 '기본 모델 정확도를 높여야 사슬 안 붙는 부분도 줄겠지 -> 진행해').  서술(판정 없음), 실행 전 커밋.  온도.
CV8·N·CK2·EL1 (FL7 거울 표) 에서 '어제 정보 없는 기록' = K8(p >= .8, dT <= 1.5, 정답 전날)도 CH5 사슬(p >= .5, dT <= 1.5, 정답 없는 전날)도 아닌 기록만.
구성원: BASE(2시드 평균), CODEX, PFN(A·B 평균), 셀 W40G-S(4셀 평균), +GRU10(엄격 10/12), +GRU30(8/10).  오차를 기록 하루 평균(수준) + 하루 안(모양)으로 나눔.
출력: [1] 구성원별 전체·수준·모양 RMSE  [2] 모양 RMSE 시각대 x 구성원  [3] 모양 RMSE 실내온도 구간 x 구성원  [4] 시각대별 '모양' 최선 볼록 섞기(BASE/CODEX/PFN .1 격자, 서술 상한)
  [5] 모양 오차 상관(구성원 간) — 섞기 여지.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u bmx1_unlinked_member_shape_v1.py
"""
import os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pch5_chain_mirror_links_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch5_chain_mirror_links_v1.py")', '__file__ = os.path.join(HERE, "bmx1_unlinked_member_shape_v1.py")')
exec(compile(src[:src.index("src2 = open(os.path.join(HERE, \"tv3_v3_gru_on_v23_v1.py\")")], "pch5pre", "exec"))
MIR7 = pd.read_csv(os.path.join(env.LOCAL, "fl7_mirror_links.csv")); ROWS = []
for L, ckb, ckg, folds, mk in (("CV8", C8.CKB, C8.CKG, C8.layout_folds(W, lab)["CV8"], C8.masks), ("N", O1.CKB, O1.CKG, LF["N"], O1.masks),
                                ("CK2", O1.CKB, O1.CKG, LF["CK2"], O1.masks), ("EL1", CKR, O1.CKG, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckb, f)) for f in sorted(os.listdir(ckb)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(ckg, f)) for f in sorted(os.listdir(ckg)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TRS, VF = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs; VF[r] = k
    LK = {r: (p, q) for r, p, q in zip(MIR7[MIR7.layout == L].rec, MIR7[MIR7.layout == L].prev, MIR7[MIR7.layout == L].p)}; linked = set()
    for r in recs:
        if r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if not np.isfinite(dt) or dt > 1.5: continue
        if r in TRS and p in TRS[r]:
            if q >= .8 and np.isfinite(ST[IDX[p], 23]): linked.add(r)
        elif q >= .5 and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and VF.get(p) == VF.get(r): linked.add(r)
    v = M.V3LF.values if "V3LF" in M else np.full(len(M), np.nan); ok = ~np.isnan(v); vv = np.nan_to_num(v)
    cells = np.mean([B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"], 0)
    F = pd.DataFrame(dict(layout=L, rec=key, h=h, t=M.in_temp.values, y=M.sub_temp.values, BASE=M[[c for c in M.columns if c.startswith("base_R_")]].mean(1).values,
                          CODEX=M.codex_R.values, PFN=(M.pfn_A.values + M.pfn_B.values) / 2, CELLS=cells, GRU10=cells + .10 * gate(M, 10, 12) * ok * (vv - cells), GRU30=cells + .30 * gate(M, 8, 10) * ok * (vv - cells)))
    F = F[~np.isin(key, list(linked))]; ROWS.append(F); print(L, "unlinked records", F.rec.nunique(), "/", len(recs), flush=True)
D = pd.concat(ROWS, ignore_index=True); MEM = ["BASE", "CODEX", "PFN", "CELLS", "GRU10", "GRU30"]
for m in MEM:
    D["e_" + m] = D[m] - D.y; g = D.groupby(["layout", "rec"])["e_" + m]; D["l_" + m] = g.transform("mean"); D["s_" + m] = D["e_" + m] - D["l_" + m]
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); pd.set_option("display.width", 220)
print("\n[1] member RMSE: total / level / shape (unlinked records, all layouts)")
print(pd.DataFrame([dict(member=m, total=rm(D["e_" + m]), level=rm(D["l_" + m]), shape=rm(D["s_" + m])) for m in MEM]).round(4).to_string(index=False))
D["hg"] = pd.cut(D.h, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]); D["band"] = pd.cut(D.t, [-99, 8, 12, 16, 20, 99], labels=["<=8", "8-12", "12-16", "16-20", ">20"])
for col, nm in (("hg", "[2] shape RMSE by hour block"), ("band", "[3] shape RMSE by in_temp band")):
    T = D.groupby(col, observed=True).apply(lambda g: pd.Series({m: rm(g["s_" + m]) for m in MEM} | {"rows": len(g)})); print(f"\n{nm}"); print(T.round(4).to_string())
print("\n[4] best convex mix on SHAPE error per hour block (descriptive upper bound; .1 grid of BASE/CODEX/PFN) vs CELLS shape")
GR = [(a, b) for a in np.round(np.arange(0, 1.01, .1), 1) for b in np.round(np.arange(0, 1.01, .1), 1) if a + b <= 1.0001]
for hb, g in D.groupby("hg", observed=True):
    sb, sc, sp = g.s_BASE.values, g.s_CODEX.values, g.s_PFN.values
    best = min(GR, key=lambda w: rm(w[0] * sb + w[1] * sc + (1 - w[0] - w[1]) * sp))
    print(f"  {hb}: best B{best[0]}/C{best[1]}/P{1 - best[0] - best[1]:.1f} shape {rm(best[0] * sb + best[1] * sc + (1 - best[0] - best[1]) * sp):.4f} vs CELLS {rm(g.s_CELLS):.4f}")
print("\n[5] correlation of shape errors between members"); print(D[["s_" + m for m in ("BASE", "CODEX", "PFN", "GRU30")]].corr().round(3).to_string())
D.to_csv(os.path.join(HERE, "bmx1_rows_v1.csv.gz"), index=False, compression="gzip")
