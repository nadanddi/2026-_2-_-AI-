# -*- coding: utf-8 -*-
"""CR4 (2026-10-13 집 클로드; 사용자: '에포크 바꿔봐').  사전 고정(커밋 후 실행).  온도.
CR1(같은 동 사슬 GRU, 80 에폭 고정, 조기 종료 없음 → G1 DIAG10 +2.2%·EXT10 +7.8% FAIL) 의 실패가 에폭(과적합) 때문인지 확인.
모델·입력·학습원(DIAG10 OOF, 퍼지)·기준(14회차형)·판정 규칙은 CR1 그대로, 학습 길이만 바꿈.
판정 안(2개, α .0125):  G1-ES, G2-ES = 폴드마다 학습 기록을 농장×10일 덩어리로 나눠 20% 를 내부 검증으로 떼고,
  내부 검증 손실이 가장 낮은 에폭의 가중치 사용(최대 80, 10 에폭 개선 없으면 멈춤), 시드 1/2/3 평균.  에폭은 학습 데이터 안에서만 정함.
서술(판정 아님 — 여기서 고른 에폭을 채택하지 않음): 내부 검증 없이 전체 학습 기록으로 학습하며 5/10/20/40/80 에폭 시점 예측 → DIAG10·EXT10 곡선.
판정(온도 범위 ALL, CR1 과 같음): 주 DIAG10·EXT10 모두 4칸 개선 & 블록 부트스트랩 P(악화) < .0125; 나머지 배치는 칸 평균 악화 & P(후보 나음) < .0125 면 FAIL.
"""
import os, sys, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(src[:src.index("def fit_predict")])
exec(src[src.index("res = []; blocks = {}"):src.index("for L, (M, cells) in FR.items():")])   # FR, layout_maps, DREC, DQ, DA, DY, purge
SNAP = (5, 10, 20, 40, 80)
def _z(Xtr):
    mu = np.nanmean(Xtr.reshape(-1, Xtr.shape[2]), 0); sd = np.nanstd(Xtr.reshape(-1, Xtr.shape[2]), 0) + 1e-6
    for flag in (len(V) + 1, len(V) + 2): mu[flag] = 0; sd[flag] = 1
    return lambda A: torch.tensor(np.nan_to_num((A - mu) / sd).astype(np.float32)).to(DEV)
def _loss(net, x, y, m): out = net(x)[:, 24:]; return ((out - y) ** 2 * m).sum() / m.sum().clamp(min=1)
def fit_all(tr, Xtr, Ytr, Xte):
    """returns {'ES': pred, 5: pred, 10: ...} and list of chosen epochs"""
    z = _z(Xtr); xt, xe = z(Xtr), z(Xte); yt = torch.tensor(np.nan_to_num(Ytr).astype(np.float32)).to(DEV); mt = torch.tensor(np.isfinite(Ytr).astype(np.float32)).to(DEV)
    blk = np.array([f"{FARM[idx[r]]}_{int(NUM[idx[r]]) // 10}" for r in tr]); ub = np.unique(blk)
    vb = set(np.random.default_rng(777).choice(ub, max(1, int(round(.2 * len(ub)))), replace=False)); iv = np.isin(blk, list(vb))
    ti, vi = torch.tensor(np.where(~iv)[0]).to(DEV), torch.tensor(np.where(iv)[0]).to(DEV)
    out = {k: [] for k in ("ES",) + SNAP}; chosen = []
    for seed in (1, 2, 3):
        # (a) snapshots on all training records
        torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(1, 81):
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), 64):
                j = perm[i:i + 64]; loss = _loss(net, xt[j], yt[j], mt[j]); opt.zero_grad(); loss.backward(); opt.step()
            if ep in SNAP:
                with torch.no_grad(): out[ep].append(net(xe)[:, 24:].cpu().numpy())
        # (b) early stopping on inner block hold-out
        torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4); g = torch.Generator(device="cpu").manual_seed(seed)
        best, best_ep, best_state, wait = np.inf, 0, copy.deepcopy(net.state_dict()), 0
        for ep in range(1, 81):
            perm = ti[torch.randperm(len(ti), generator=g).to(DEV)]
            for i in range(0, len(perm), 64):
                j = perm[i:i + 64]; loss = _loss(net, xt[j], yt[j], mt[j]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): vl = float(_loss(net, xt[vi], yt[vi], mt[vi]))
            if vl < best - 1e-6: best, best_ep, best_state, wait = vl, ep, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= 10: break
        net.load_state_dict(best_state); chosen.append(best_ep)
        with torch.no_grad(): out["ES"].append(net(xe)[:, 24:].cpu().numpy())
    return {k: np.mean(v, 0) for k, v in out.items()}, chosen
ARMS = ["ES"] + list(SNAP); EPS = []
for L, (M, cells) in FR.items():
    key, h, y, anc, recs, pos, Q, A, Y = layout_maps(M, cells)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; corr = {(b, a): {} for b in ("G2", "G1") for a in ARMS}
    for k in sorted(set(fold_of.values())):
        te = [r for r in recs if fold_of[r] == k]; hid = set(te); tr = [r for r in DREC if r not in purge(te)]
        for b, base, dbase in (("G2", Q, DQ), ("G1", A, DA)):
            Xtr = build(tr, dbase, hid); Ytr = np.stack([DY[r] - dbase[r] for r in tr]); Xte = build(te, base, hid)
            P, ch = fit_all(tr, Xtr, Ytr, Xte); EPS.append(dict(layout=L, fold=k, base=b, ep=ch))
            for a in ARMS:
                for r, p in zip(te, P[a]): corr[(b, a)][r] = p
    for (b, a), cmap in corr.items():
        for ci in range(len(cells)):
            src_ = cells[ci] if b == "G2" else anc[ci]; new = src_.copy(); base = anc[ci]
            for r in recs: ix = pos[r]; new[ix] = src_[ix] + cmap[r][h[ix]]
            p2 = M.day.values >= 179; nm = f"{b}-{a}"
            res.append(dict(layout=L, cell=ci, arm=nm, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, nm)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
E = pd.DataFrame(EPS); allep = [e for l in E.ep for e in l]
print("early-stopping chosen epochs: median", np.median(allep), "| quantiles", np.percentile(allep, [10, 25, 75, 90]).tolist(), "| by base", E.groupby("base").ep.apply(lambda s: float(np.median([e for l in s for e in l]))).to_dict())
S = pd.DataFrame(res); pd.set_option("display.width", 240)
print("\n[descriptive] epoch curve, cell-mean pct vs 14th-type:")
print(S[S.layout.isin(["DIAG10", "EXT10"])].groupby(["arm", "layout"]).pct.mean().unstack().round(3).to_string())
print("\n[judged arms] per-cell pct:"); print(S[S.arm.str.endswith("-ES")].pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(S[S.arm.str.endswith("-ES")].groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("G2-ES", "G1-ES"):
    ok = True; msg = []
    for L in sorted(set(S.layout)):
        cells_ = S[(S.arm == a) & (S.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10"):
            c = bool((cells_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((cells_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = cells_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {cells_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== CR4 {a}:"); print("\n".join("   " + m for m in msg)); print(f"CR4 {a}:", "ADOPT" if ok else "FAIL")
