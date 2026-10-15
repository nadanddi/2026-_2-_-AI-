# -*- coding: utf-8 -*-
"""CH8 (2026-10-16 집 클로드; 사용자 '진행해' — 새 정보원 없이 이미 가진 신호로 진짜/가짜 연결 가르기).  서술(판정 없음), 실행 전 커밋.  온도.
CH7 후보(247개)에 검증 신호 추가: (a) 예측 연속성 |A − .36 − x0| (전날 기준 예측 23시 vs 오늘 셀 0시) (b) 다른 링커 일치 — v7 거울 CV(cv_all400_choice_v7), v10m, v15R 거울(prev0·p0, 1·2위 차), FLN(prev·p·p_none) 이 FL5 와 같은 전날을 고르나.
진짜/가짜(truth.pkl cert) 가르는 AUC(신호별·합침 교차), p < .5 구간 안에서도.  아래는 CH7 원문.
EXT10·EXT12·DIAG10 (TV3 틀, FL5 거울 표), 사슬 후보(정답 없는 같은 폴드 같은 농장 앞 날짜 전날, p >= .3, dT <= 3) 전부에 M1 사슬을 걸었을 때 기록별 dSSE(사슬 − 없음, 4셀 평균).
근거 가설: H1 '연결이 진짜 어제인가' — 정답 이어짐 표(linker_v3 truth.pkl, cert)로 진짜 전날 여부 + 정답 연속성 |배지 23시(전날) − 배지 0시(오늘)|;
          H2 '밤사이 상황이 바뀌었나' — dT(0시 실내 − 전날 23시 실내);  H3 '고칠 거리' — |A − .36 − x0|(사슬 보정 크기)와 오늘 0시 실제 오차의 같은 방향 여부.
출력: 확신 구간 x (진짜 전날 비율, 배지 연속성 중앙, 기록 dSSE 합·평균, 도움/해), 진짜/가짜 연결별 dSSE, dT 구간별, 보정 방향 맞음 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ch7_why_p_band_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ch6_chain_p05_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch5_chain_mirror_links_v1.py")', '__file__ = os.path.join(HERE, "ch8_link_verification_signals_v1.py")')
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
        cells0 = np.mean([B2.w40(M, b, cx, fam) for b in bs for fam in 'AB'], 0); b0m = np.mean(base, 0); ip_ = pos[p]; urw = b0m[ip_][h[ip_] == 23][0] - .36 - cells0[ix][h[ix] == 0][0]; x0e = b0m[ix][h[ix] == 0][0] - y[ix][h[ix] == 0][0]; corr_dir = np.sign((np.mean(ch, 0) - b0m)[ix][h[ix] == 0][0]) == -np.sign(x0e)
        ip = pos[p]; cont = abs(y[ip][h[ip] == 23][0] - y[ix][h[ix] == 0][0])
        ROWS.append(dict(layout=L, rec=r, prev=p, p=q, dt=dt, dsse=dd, true_known=r in TRUEP, true_prev=(TRUEP.get(r) == p) if r in TRUEP else np.nan, sub_cont=cont, dir_ok=bool(corr_dir), err0=abs(x0e), u_raw=urw))
    print(L, "candidates", len(CAND), flush=True)
D = pd.DataFrame(ROWS); D.to_csv(os.path.join(HERE, "ch8_candidates_v1.csv"), index=False); D["pbin"] = pd.cut(D.p, [.29, .5, .65, .8, 1.01]); D["dtbin"] = pd.cut(D.dt, [-.01, .5, 1.5, 3.01]); pd.set_option("display.width", 220)
agg = lambda g: pd.Series(dict(n=len(g), true_rate=g.true_prev.dropna().mean() if g.true_prev.notna().any() else np.nan, n_known=int(g.true_known.sum()), sub_cont_med=g.sub_cont.median(),
                               dsse_sum=g.dsse.sum(), dsse_mean=g.dsse.mean(), helped=int((g.dsse < -1e-9).sum()), harmed=int((g.dsse > 1e-9).sum()), dir_ok=g.dir_ok.mean()))
print("\n[p band]"); print(D.groupby("pbin", observed=True).apply(agg).round(3).to_string())
print("\n[dT band]"); print(D.groupby("dtbin", observed=True).apply(agg).round(3).to_string())
print("\n[true vs false previous day (known only)]"); print(D[D.true_known].assign(tp=lambda x: x.true_prev.astype(bool)).groupby("tp").apply(agg).round(3).to_string())
D["cont_bin"] = pd.cut(D.sub_cont, [-.01, .5, 1.0, 2.0, 99]); print("\n[substrate continuity |sub23(prev) - sub0(rec)| (label-based, diagnosis only)]"); print(D.groupby("cont_bin", observed=True).apply(agg).round(3).to_string())
D.to_csv(os.path.join(HERE, "ch8_candidates_v1.csv"), index=False)

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7.csv"), encoding="utf-8-sig").set_index("rec")
V10 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "cv_all400_choice_v10m.csv"), encoding="utf-8-sig").set_index("rec")
V15 = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")).set_index("rec"); FLN = pd.read_csv(os.path.join(env.LOCAL, "fln1_train_links.csv")).set_index("rec")
D["agree_v7"] = [float(V7.prev.get(r) == p) for r, p in zip(D.rec, D.prev)]; D["agree_v10"] = [float(V10.prev.get(r) == p) for r, p in zip(D.rec, D.prev)]
D["agree_v15"] = [float(V15.prev0.get(r) == p) for r, p in zip(D.rec, D.prev)]; D["v15_margin"] = [V15.p0.get(r, np.nan) - V15.p1.get(r, np.nan) for r in D.rec]
D["agree_fln"] = [float(FLN.prev.get(r) == p) for r, p in zip(D.rec, D.prev)]; D["fln_pnone"] = [FLN.p_none.get(r, np.nan) for r in D.rec]
D["n_agree"] = D[["agree_v7", "agree_v10", "agree_v15", "agree_fln"]].sum(1); D["abs_u"] = D.u_raw.abs()
K = D[D.true_known].copy(); yk = K.true_prev.astype(bool).astype(int).values
print(f"\n[signals] known links {len(K)} (true {yk.sum()}, false {len(yk) - yk.sum()})")
for f in ("p", "dt", "abs_u", "agree_v7", "agree_v10", "agree_v15", "v15_margin", "agree_fln", "fln_pnone", "n_agree"):
    v = K[f].fillna(K[f].median()).values; a = roc_auc_score(yk, v) if len(set(v)) > 1 else .5
    print(f"  {f:11s} AUC(true) {a:.3f} | true mean {K[f][yk == 1].mean():.3f} vs false {K[f][yk == 0].mean():.3f}")
FEATS = ["p", "dt", "abs_u", "n_agree", "v15_margin", "fln_pnone"]; Z = K[FEATS].fillna(K[FEATS].median()).values; pr = np.zeros(len(K))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, yk): pr[te] = LogisticRegression(C=1, max_iter=2000).fit(Z[tr], yk[tr]).predict_proba(Z[te])[:, 1]
print(f"  combined CV AUC {roc_auc_score(yk, pr):.3f} (p alone {roc_auc_score(yk, K.p):.3f})")
print("\n[within p < .5 candidates] chain dSSE by number of other linkers agreeing / |u| band")
S = D[D.p < .5]
print(S.groupby("n_agree").dsse.agg(["size", "sum", "mean"]).round(3).to_string())
S = S.assign(ub=pd.cut(S.abs_u, [-.01, .5, 1.0, 99])); print(S.groupby("ub", observed=True).dsse.agg(["size", "sum", "mean"]).round(3).to_string())
print("\n[all candidates] chain dSSE by n_agree x p band")
print(D.assign(pb=pd.cut(D.p, [.29, .5, .8, 1.01])).groupby(["pb", "n_agree"], observed=True).dsse.agg(["size", "sum", "mean"]).round(3).to_string())
D.to_csv(os.path.join(HERE, "ch8_candidates_v1.csv"), index=False)
