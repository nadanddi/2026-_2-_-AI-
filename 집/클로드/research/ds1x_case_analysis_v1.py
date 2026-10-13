# -*- coding: utf-8 -*-
"""DS1X (2026-10-13 집 클로드; 사용자: 'HMM 이 잘 맞추는 경우와 못 맞추는 경우 확인').  서술(판정 아님).  EC.
DS1(6.806) 그대로(DIAG10 3시드, 상한 사슬) 다시 돌려 기록마다: 실제 상태(하루 평균 ≥ 1.0), 예측 하루 평균, 23시 사후 P(H), 사슬 길이,
사슬 이전 날의 실제 상태·예측, 오늘 z, 보정 평균 이동, ΔSSE(시드 평균).  이득/손해 기록을 유형별로 묶음.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DS1SRC = open(os.path.join(HERE, "ds1_day_state_hmm_ec_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ds1_day_state_hmm_ec_v1.py")
exec(DS1SRC[:DS1SRC.index("res = []; recrows = []")])
rows = []
for sd, G in O.groupby("seed"):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values
    G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); cm = G.cm.values
    day_sg = G.groupby("rec").sg.mean(); hidden = G.groupby("fold").rec.apply(set).to_dict()
    for k, hs in hidden.items():
        tr = [r for r in day_sg.index if r not in hs]; st = {r: DM[idx[r]] >= 1.0 for r in tr}
        vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
        pr = [(st[PREV[r]], st[r]) for r in tr if r in PREV and PREV[r] in st]
        A = np.array([[sum(1 for a, b in pr if a == s0 and b == s1) + 1 for s1 in (0, 1)] for s0 in (0, 1)], float); A /= A.sum(1, keepdims=True)
        prior = np.array([1 - np.mean(list(st.values())), np.mean(list(st.values()))])
        cs = np.array([day_sg[r] for r in tr]); ss = np.array([st[r] for r in tr]).astype(int); ys = np.array([DM[idx[r]] for r in tr])
        zs = np.array([zflag(idx[r], 23, vis) for r in tr])
        mu = [cs[ss == s].mean() for s in (0, 1)]; sdv = [max(cs[ss == s].std(), .05) for s in (0, 1)]; pi = [(zs[ss == s].sum() + 1) / ((ss == s).sum() + 2) for s in (0, 1)]
        reg = [np.polyfit(cs[ss == s], ys[ss == s], 1) for s in (0, 1)]
        lik = lambda c, z: np.array([norm.pdf(c, mu[s], sdv[s]) * (pi[s] if z else 1 - pi[s]) for s in (0, 1)])
        for r in hs:
            if r not in day_sg.index: continue
            ix = G.index[G.rec == r].values; i = idx[r]
            chain = []; cur = r
            for _ in range(3):
                pv = PREV.get(cur)
                if pv is None or pv not in day_sg.index: break
                chain.append(pv); cur = pv
            alpha = prior.copy()
            for pv in reversed(chain):
                alpha = alpha @ A; alpha = alpha * lik(day_sg[pv], zflag(idx[pv], 23, vis)); alpha /= alpha.sum()
            a0 = (alpha @ A) if chain else prior; new = sg[ix].copy(); pH = np.nan
            for n_, j in enumerate(ix):
                h = int(G.hour.values[j]); c = cm[j]; z = zflag(i, h, vis)
                post = a0 * lik(c, z); post /= post.sum(); pH = post[1]
                yh = sum(post[s] * np.polyval(reg[s], c) for s in (0, 1)); new[n_] = np.clip(sg[j] + (yh - c), lo_, hi_)
            rows.append(dict(seed=sd, rec=r, p2=NUM[i] >= 179, trueH=DM[i] >= 1.0, y=DM[i], pred=sg[ix].mean(), shift=(new - sg[ix]).mean(), pH23=pH,
                             nchain=len(chain), prev_trueH=(DM[idx[chain[0]]] >= 1.0) if chain else None, prev_pred=day_sg[chain[0]] if chain else np.nan,
                             z23=zflag(i, 23, vis), dsse=float(np.sum((new - y[ix]) ** 2) - np.sum((sg[ix] - y[ix]) ** 2))))
D = pd.DataFrame(rows).groupby("rec").agg(p2=("p2", "first"), trueH=("trueH", "first"), y=("y", "first"), pred=("pred", "mean"), shift=("shift", "mean"), pH23=("pH23", "mean"),
                                          nchain=("nchain", "first"), prev_trueH=("prev_trueH", "first"), prev_pred=("prev_pred", "mean"), z23=("z23", "first"), dsse=("dsse", "mean")).reset_index()
D.to_csv(os.path.join(HERE, "ds1x_case_rows_v1.csv"), index=False); pd.set_option("display.width", 240)
D["predH"] = D.pred >= 1.0; D["chain"] = np.where(D.nchain == 0, "none", np.where(D.prev_trueH == True, "prevH", "prevN"))  # noqa: E712
D["type"] = np.select([D.trueH & D.predH, D.trueH & ~D.predH, ~D.trueH & D.predH], ["H_seen", "H_missed", "N_falseH"], "N_ok")
print(f"records {len(D)} | total dSSE {D.dsse.sum():+.2f} (neg = better) | pass-2 {D[D.p2].dsse.sum():+.2f}")
print("\n[by type x chain] size / sum dSSE / mean shift / mean P(H)")
print(D.groupby(["type", "chain"]).agg(n=("rec", "size"), dsse=("dsse", "sum"), shift=("shift", "mean"), pH=("pH23", "mean")).round(3).to_string())
print("\n[by pass] "); print(D.groupby(["p2", "type"]).agg(n=("rec", "size"), dsse=("dsse", "sum")).round(3).to_string())
print("\n[top 10 gains]"); print(D.sort_values("dsse").head(10)[["rec", "type", "chain", "y", "pred", "prev_pred", "pH23", "shift", "z23", "dsse"]].round(3).to_string(index=False))
print("\n[top 10 losses]"); print(D.sort_values("dsse").tail(10)[["rec", "type", "chain", "y", "pred", "prev_pred", "pH23", "shift", "z23", "dsse"]].round(3).to_string(index=False))
print("\nconcentration: share of total gain from top 5 records", round(float(D[D.dsse < 0].nsmallest(5, "dsse").dsse.sum() / D[D.dsse < 0].dsse.sum()), 3),
      "| share of total loss from top 3", round(float(D[D.dsse > 0].nlargest(3, "dsse").dsse.sum() / D[D.dsse > 0].dsse.sum()), 3))
