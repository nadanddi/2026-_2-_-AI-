# -*- coding: utf-8 -*-
"""EV3L (2026-10-15 집 클로드; 사용자 취침 중 '뭐라도 마구 해봐').  사전 고정(커밋 후 실행).  EC.
온도 V3L(6.912 ADOPT): 번호 순서 창 → 링커로 찾은 같은 동 전날 창으로 바꾸니 이득.  EC 하루 GRU(EG1 6.830, MT1 A)는 자기 하루만 봐서 P2LOO 에서 약했음.
EV3L = EC 2층 GRU(128)에 48시간 창: [링커 짝 전날 0..23 입력] + [오늘 0..23 입력], 출력은 오늘 위치 24개(단방향 → h 시는 전날 + 오늘 0..h 만).
  링커 = v15R 거울 prev0, p0 ≥ .5(V3L 과 같음); 짝 없으면 전날 위치 NaN(→ 0 + 결측 표시).  입력·표준화·학습(Adam 2e-3, 25 에폭, 배치 16, 내부 시드 1·2) = MT1 A.
  기준 판 E0 = 같은 망·같은 학습의 오늘 24시간만(전날 위치 전부 NaN) — 전날 창 효과만 분리.
섞기: 0.9·sg + 0.1·X (EG1 과 같은 고정 비율), ec_rv1 CUR(DIAG10·EL1·P2LOO, 시드 47·1414·6464), 숨김 = MT1 과 같음.
판정(EC 분야 배정 규칙, EV3L 섞기 vs E0 섞기, k 1): (1) P2LOO·EL1 모든 시드 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
  [링커를 쓰므로 EV 견고성이 원래 필요 — ec_rv1 에 EV 없음, 한계로 보고]
서술: 단독 RMSE(EV3L / E0 / sg), sg 대비 섞기 %, 짝 있는 기록 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ev3l_ec_linked_yesterday_gru_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("def fit_predict")])
__file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
PMx = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); LINK = {r: p for r, p, q in zip(PMx.rec, PMx.prev0, PMx.p0) if isinstance(p, str) and q >= .5 and p != r and p in idx}
C48 = np.full((N, 48, CUBE.shape[2]), np.nan, np.float32); C48[:, 24:] = CUBE
for r, p in LINK.items():
    if r in idx: C48[idx[r], :24] = CUBE[idx[p]]
C00 = C48.copy(); C00[:, :24] = np.nan
print("links", len(LINK), "| labelled records with yesterday window", float(np.mean([r in LINK for r in LABREC])), flush=True)
def fit48(tr, te, CC):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A_ = CC[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CC[I] - mu) / sd; m = np.isnan(CC[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV); out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = NetA(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; e = net(xt[j])[0][:, 24:]; mj = mt[j]
                loss = (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe)[0][:, 24:].cpu().numpy() + ym)
    return np.mean(out, 0)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
PRED = {"EV3L": {}, "E0": {}}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind, CC in (("EV3L", C48), ("E0", C00)):
            p = fit48(tr, te, CC)
            for j, r in enumerate(te): PRED[kind][(v, r)] = p[j]
    print("done", v, flush=True)
res, B = [], []
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    m1 = np.array([PRED["EV3L"][(v, r)][h] for r, h in zip(G.rec, G.hour)]); m0 = np.array([PRED["E0"][(v, r)][h] for r, h in zip(G.rec, G.hour)])
    b1, b0 = .9 * sg + .1 * m1, .9 * sg + .1 * m0
    B.append(pd.DataFrame(dict(val=v, seed=sd_, rec=G.rec, farm=G.rec.str[:3], day=G.day, y=y, d=(b1 - y) ** 2 - (b0 - y) ** 2, p2=p2)))
    res.append(dict(val=v, seed=sd_, alone_EV3L=rm(m1 - y), alone_E0=rm(m0 - y), alone_sg=rm(sg - y), dSSE_p2=float(((b1 - y) ** 2 - (b0 - y) ** 2)[p2].sum()),
                    blend_vs_sg_EV3L=100 * (rm(b1 - y) / rm(sg - y) - 1), blend_vs_sg_E0=100 * (rm(b0 - y) / rm(sg - y) - 1)))
S = pd.DataFrame(res); B = pd.concat(B, ignore_index=True); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
def boot(v, p2only):
    g = B[(B.val == v) & (B.p2 if p2only else True)]; s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20261230).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot("P2LOO", True)
RD = B[B.val.isin(["P2LOO", "EL1"]) & B.p2].groupby(["val", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum())
_, pb = boot("DIAG10", False); c3 = not (B[B.val == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
print(f"\n(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("EV3L:", "ADOPT" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
