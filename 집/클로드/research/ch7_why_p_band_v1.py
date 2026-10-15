# -*- coding: utf-8 -*-
"""CH7 (2026-10-16 집 클로드; 사용자 '왜 확신 .5~.8 때만 효과적인지 근거 확인').  서술(판정 없음), 실행 전 커밋.  온도.
EXT10·EXT12·DIAG10 (TV3 틀, FL5 거울 표), 사슬 후보(정답 없는 같은 폴드 같은 농장 앞 날짜 전날, p >= .3, dT <= 3) 전부에 M1 사슬을 걸었을 때 기록별 dSSE(사슬 − 없음, 4셀 평균).
근거 가설: H1 '연결이 진짜 어제인가' — 정답 이어짐 표(linker_v3 truth.pkl, cert)로 진짜 전날 여부 + 정답 연속성 |배지 23시(전날) − 배지 0시(오늘)|;
          H2 '밤사이 상황이 바뀌었나' — dT(0시 실내 − 전날 23시 실내);  H3 '고칠 거리' — |A − .36 − x0|(사슬 보정 크기)와 오늘 0시 실제 오차의 같은 방향 여부.
출력: 확신 구간 x (진짜 전날 비율, 배지 연속성 중앙, 기록 dSSE 합·평균, 도움/해), 진짜/가짜 연결별 dSSE, dT 구간별, 보정 방향 맞음 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ch7_why_p_band_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ch6_chain_p05_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch5_chain_mirror_links_v1.py")', '__file__ = os.path.join(HERE, "ch7_why_p_band_v1.py")')
exec(compile(src[:src.index("MIR = pd.read_csv(os.path.join(env.LOCAL, \"fl5_mirror_links.csv\")); RES = {}")], "ch6pre", "exec"))
TT = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl")); TT = TT[TT.cert]
TRUEP = {f"{fb}_{int(nb):03d}": f"{fa}_{int(na):03d}" for fb, nb, fa, na in zip(TT.fb, TT.nb, TT.fa, TT.na)}
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); ROWS = []
for L in ("EXT10", "EXT12", "DIAG10"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); M = M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    TXo = tx3(M, key, .25, 10, .8); FO = dict(zip(key, M.fold.values)); HELD = {}
    for r, k in FO.items(): HELD.setdefault(k, set()).add(r)
    LK = {r: (p, q) for r, p, q in zip(MIR[MIR.layout == L].rec, MIR[MIR.layout == L].prev, MIR[MIR.layout == L].p)}; order = sorted(recs, key=lambda r: (r[:3], int(r[4:7])))
    AF, _ = chains_t(LK, recs, pos, h, FO, lambda r, p: p not in HELD[FO[r]], .8)
    CAND = {}
    for r in recs:
        if r in AF or r not in LK or not isinstance(LK[r][0], str): continue
        p, q = LK[r]
        if p not in IDX or q < .3 or p in AF and False: continue
        dt = abs(IN_[IDX[r], 0, 0] - IN_[IDX[p], 23, 0])
        if np.isfinite(dt) and dt <= 3 and p in HELD[FO[r]] and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) and p in pos and (h[pos[p]] == 23).any(): CAND[r] = (p, q, dt)
    run0 = make_run(AF, {}, pos, h, TXo, order)
    base = [run0(B2.w40(M, b, cx, fam), None) for b in bs for fam in "AB"]
    for r, (p, q, dt) in CAND.items():
        runr = make_run(AF, {r: p}, pos, h, TXo, order)
        ch = [runr(B2.w40(M, b, cx, fam), lambda ix: S1 * np.exp(-h[ix] / TAU)) for b in bs for fam in "AB"]
        ix = pos[r]; dd = np.mean([((a - y) ** 2 - (b0 - y) ** 2)[ix].sum() for a, b0 in zip(ch, base)])
        b0m = np.mean(base, 0); x0e = b0m[ix][h[ix] == 0][0] - y[ix][h[ix] == 0][0]; corr_dir = np.sign((np.mean(ch, 0) - b0m)[ix][h[ix] == 0][0]) == -np.sign(x0e)
        ip = pos[p]; cont = abs(y[ip][h[ip] == 23][0] - y[ix][h[ix] == 0][0])
        ROWS.append(dict(layout=L, rec=r, prev=p, p=q, dt=dt, dsse=dd, true_known=r in TRUEP, true_prev=(TRUEP.get(r) == p) if r in TRUEP else np.nan, sub_cont=cont, dir_ok=bool(corr_dir), err0=abs(x0e)))
    print(L, "candidates", len(CAND), flush=True)
D = pd.DataFrame(ROWS); D["pbin"] = pd.cut(D.p, [.29, .5, .65, .8, 1.01]); D["dtbin"] = pd.cut(D.dt, [-.01, .5, 1.5, 3.01]); pd.set_option("display.width", 220)
agg = lambda g: pd.Series(dict(n=len(g), true_rate=g.true_prev.dropna().mean() if g.true_prev.notna().any() else np.nan, n_known=int(g.true_known.sum()), sub_cont_med=g.sub_cont.median(),
                               dsse_sum=g.dsse.sum(), dsse_mean=g.dsse.mean(), helped=int((g.dsse < -1e-9).sum()), harmed=int((g.dsse > 1e-9).sum()), dir_ok=g.dir_ok.mean()))
print("\n[p band]"); print(D.groupby("pbin", observed=True).apply(agg).round(3).to_string())
print("\n[dT band]"); print(D.groupby("dtbin", observed=True).apply(agg).round(3).to_string())
print("\n[true vs false previous day (known only)]"); print(D[D.true_known].groupby("true_prev").apply(agg).round(3).to_string())
D["cont_bin"] = pd.cut(D.sub_cont, [-.01, .5, 1.0, 2.0, 99]); print("\n[substrate continuity |sub23(prev) - sub0(rec)| (label-based, diagnosis only)]"); print(D.groupby("cont_bin", observed=True).apply(agg).round(3).to_string())
D.to_csv(os.path.join(HERE, "ch7_candidates_v1.csv"), index=False)
