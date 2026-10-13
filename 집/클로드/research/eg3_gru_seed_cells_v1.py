# -*- coding: utf-8 -*-
"""EG3 (2026-10-14 집 클로드; 비평 G1).  사전 고정(커밋 후 실행).  EC.  자기 기록 입력만.
비평(6.870): EG1 의 '9/9 칸'은 GRU 하나(초기화 시드 1·2 평균)를 세 기본 시드에 재사용 → 기본 시드 견고성일 뿐 GRU 견고성이 아님.
EG3 = EG1 과 똑같은 GRU(2층·128, 25 에폭, 배치 16, Adam 2e-3, 같은 입력·표준화·숨김 범위), 단 GRU 초기화 시드를 새로 31·32·33 하나씩
  각각 칸 하나로: (기본 시드 47, GRU 31), (1414, 32), (6464, 33).  후보 = 0.9 · sg + 0.1 · GRU (비율 고정, EG1 과 같음).
판정(EG1 과 같은 EC 분야 배정 규칙, k = 1): (1) P2LOO·EL1 모든 칸 ΔSSE < 0 AND (P2LOO 칸 평균 농장×5일 블록 P(악화) < .025 OR 나빠진 기록 0);
  (3) DIAG10 손해 감시.  [한계: 같은 검증기 재사용 — 새 정보는 GRU 무작위성뿐]
서술: 소진 기록 장부(F13_231·F13_233·F47_216·F47_229·F47_241·F13_217·F13_243·F47_231) 제외 ΔSSE,
  평가 노출 = 전체 학습 정답 기록으로 GRU(31·32·33 평균) 학습 → 0.9 · 14회차 EC + 0.1 · GRU 의 평가 기록별 하루 평균 Δ(특히 14회차 >= 1 인 기록).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py")
exec(src[:src.index("def fit_predict")])
__file__ = os.path.join(HERE, "eg3_gru_seed_cells_v1.py")
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
def fit_predict(tr, te, seed):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A = CUBE[it]; mu = np.nanmean(A.reshape(-1, A.shape[2]), 0); sd = np.nanstd(A.reshape(-1, A.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
    for ep in range(25):
        perm = torch.randperm(len(xt), generator=g).to(DEV)
        for b in range(0, len(xt), 16):
            j = perm[b:b + 16]; loss = (((net(xt[j]) - yt[j]) ** 2) * mt[j]).sum() / mt[j].sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xe).cpu().numpy() + ym
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
BSEEDS = sorted(O.seed.unique()); PAIR = dict(zip(BSEEDS, (31, 32, 33))); print("cells (base seed -> GRU seed):", PAIR, flush=True)
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
GRU = {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for gs in PAIR.values():
        P = {}
        for k, Gk in G.groupby("fold"):
            te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set())
            tr = [r for r in LABREC if r not in hide]; pr = fit_predict(tr, te, gs)
            for r, p in zip(te, pr): P[r] = p
        GRU[(v, gs)] = P; print("done", v, gs, len(P), flush=True)
res, blocks, recd = [], {}, []
for (v, sd), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; gs = PAIR[sd]
    gr = np.array([GRU[(v, gs)][r][h] for r, h in zip(G.rec, G.hour)]); new = .9 * sg + .1 * gr
    p2 = G.day.values >= 179; d = (new - y) ** 2 - (sg - y) ** 2; led = ~G.rec.isin(LEDGER).values
    res.append(dict(val=v, seed=sd, gru_seed=gs, dSSE=float(d.sum()), dSSE_p2=float(d[p2].sum()), dSSE_noledger=float(d[led].sum()), pct=100 * (rm(new - y) / rm(sg - y) - 1)))
    blocks.setdefault(v, []).append(pd.Series(d).groupby((G.farm + "_" + (G.day // 5).astype(str)).values).mean())
    for r, dd in pd.Series(d).groupby(G.rec.values).sum().items(): recd.append(dict(val=v, seed=sd, rec=r, dsse=dd))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
RD = pd.DataFrame(recd)
print("\nkey records (cell-mean dSSE):", RD[RD.rec.isin(LEDGER)].groupby(["val", "rec"]).dsse.mean().round(3).to_dict())
def boot(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = np.random.default_rng(20261216).integers(0, len(s), (20000, len(s))); m = s[ix].mean(1)
    return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE < 0).all()); pP2, _ = boot("P2LOO")
harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum()); c1b = pP2 < .025 or nh == 0
print(f"\n(1) main P2LOO+EL1: cells better {int((main.dSSE < 0).sum())}/{len(main)} | P2LOO P(worse) {pP2:.4f} | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot("DIAG10"); c3 = not (dg.dSSE.mean() > 0 and pb < .025)
print(f"(3) harm guard DIAG10: mean dSSE {dg.dSSE.mean():+.3f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("EG3 GRU seed cells:", "ADOPT" if (c1 and c1b and c3) else "FAIL")
# eval exposure (descriptive)
ev = [rid(i) for i in range(N) if not TRAIN[i]]
PE = np.mean([fit_predict(LABREC, ev, gs) for gs in PAIR.values()], 0)
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14["h"] = P14.row_id.str[8:].astype(int)
gm = {(r, h): PE[j][h] for j, r in enumerate(ev) for h in range(24)}
P14["gru"] = [gm[(r, h)] for r, h in zip(P14.rec, P14.h)]; P14["new"] = .9 * P14.sub_ec + .1 * P14.gru
E = P14.groupby("rec").agg(cur=("sub_ec", "mean"), gru=("gru", "mean"), new=("new", "mean")); E["d"] = E.new - E.cur
print("\neval exposure (day means):"); print(E.round(3).sort_values("d").to_string())
print(f"rows RMS shift {np.sqrt(np.mean((P14.new - P14.sub_ec) ** 2)):.4f} | records with cur >= 1: mean d {E[E.cur >= 1].d.mean():+.3f} (n {int((E.cur >= 1).sum())}) | others mean d {E[E.cur < 1].d.mean():+.3f}")
E.to_csv(os.path.join(HERE, "eg3_eval_exposure_v1.csv"))
