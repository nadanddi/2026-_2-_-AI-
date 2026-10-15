# -*- coding: utf-8 -*-
"""EVX (2026-10-15 집 클로드; 사용자: 'HMM 이나 MTL 쪽에서 더 이득 못 보나? 실험해볼 만한 것').  사전 고정(커밋 후 실행).  EC.
HMM 계열(DS1~DS9, EH0~EH6, 6.816~6.868)은 가짜 고EC 사슬·2차 발동 부족으로 반복 실패 → 이번엔 제외.  EV3L(6.921 ADOPT: 같은 동 전날 48시간 창 GRU)이
  처음 통과한 EC 신경망이므로 그 위에 두 가지를 붙여 EV3L 과 비교(같은 ec_rv1 배치·숨김·링커 v15R 거울 p0 ≥ .5·내부 시드 1·2):
  EVM = MT1(6.894) 구조(1차 머리 [EC, 남은 하루 입력 7종] → mix → 2차 머리, 손실 EC1 + EC2 + .5(AUX1 + AUX2)) 를 48시간 창에(출력·손실은 오늘 위치)
  EVY = EV3L + 채널 2개: 어제 위치에 그 짝의 실제 EC(정답 있는 학습 기록이고 그 폴드 숨김(검증·EL1 이웃)이 아닐 때만, 아니면 NaN) + 있음 표시
판정(EC 분야 배정 규칙, 각 안의 0.9·sg + 0.1·X vs 0.9·sg + 0.1·EV3L(저장 예측 local/ev3l_preds_v1.pkl), 본페로니 k 2 → .0125):
  (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .0125 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
서술: 단독 RMSE, sg 대비 섞기 %, 어제 EC 를 받은 기록 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u evx_ev3l_mtl_and_yesterday_ec_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "evx_ev3l_mtl_and_yesterday_ec_v1.py")
F48 = np.full((N, 48, len(AIX)), np.nan, np.float32); F48[:, 24:] = FUT          # aux targets only at today's positions
def fit_x(tr, te, kind, hide):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te]); CC = C48.copy()
    if kind == "EVY":
        Y = np.full((N, 48, 2), np.nan, np.float32); Y[..., 1] = 0
        for r, p in LINK.items():
            if r in idx and p in idx and TRAIN[idx[p]] and p not in hide and np.isfinite(EC[idx[p]]).any():
                Y[idx[r], :24, 0] = EC[idx[p]]; Y[idx[r], :24, 1] = 1
        CC = np.concatenate([CC, Y], 2)
    A_ = CC[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CC[I] - mu) / sd; m = np.isnan(CC[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Yt = EC[it].astype(np.float32); ym = float(np.nanmean(Yt))
    yt = torch.tensor(np.nan_to_num(Yt - ym), device=DEV); mt = torch.tensor(np.isfinite(Yt).astype(np.float32), device=DEV)
    Fa = F48[it]; fm, fs = np.nanmean(Fa.reshape(-1, len(AIX)), 0), np.nanstd(Fa.reshape(-1, len(AIX)), 0) + 1e-6
    ft = torch.tensor(np.nan_to_num((Fa - fm) / fs).astype(np.float32), device=DEV); fmask = torch.tensor(np.isfinite(Fa).astype(np.float32), device=DEV); out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = (NetB(xt.shape[2], len(AIX)) if kind == "EVM" else NetA(xt.shape[2])).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; o = net(xt[j]); mj = mt[j]; ecl = lambda e: (((e[:, 24:] - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1)
                loss = ecl(o[0])
                if kind == "EVM":
                    fj, fmj = ft[j], fmask[j]; al = lambda a: (((a - fj) ** 2) * fmj).sum() / fmj.sum().clamp(min=1)
                    loss = loss + ecl(o[1]) + .5 * (al(o[2]) + al(o[3]))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe)[0][:, 24:].cpu().numpy() + ym)
    return np.mean(out, 0)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
PRED = {"EVM": {}, "EVY": {}}; PRED["EV3L"] = pickle.load(open(os.path.join(env.LOCAL, "ev3l_preds_v1.pkl"), "rb"))["EV3L"]; used = []
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind in ("EVM", "EVY"):
            p = fit_x(tr, te, kind, hide)
            for j, r in enumerate(te): PRED[kind][(v, r)] = p[j]
        used += [(v, r, LINK.get(r) is not None and LINK.get(r) in idx and TRAIN[idx[LINK[r]]] and LINK[r] not in hide) for r in te]
    print("done", v, flush=True)
pd.to_pickle(PRED, os.path.join(env.LOCAL, "evx_preds_v1.pkl"))
U = pd.DataFrame(used, columns=["val", "rec", "u"]); print("validation records with yesterday EC visible:", U.groupby("val").u.mean().round(3).to_dict())
pd.set_option("display.width", 220); VER = {}
for kind in ("EVM", "EVY"):
    res, B = [], []
    for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
        G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
        mx = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); m0 = np.array([PRED["EV3L"][(v, r)][h] for r, h in zip(G.rec, G.hour)])
        b1, b0 = .9 * sg + .1 * mx, .9 * sg + .1 * m0
        B.append(pd.DataFrame(dict(val=v, seed=sd_, rec=G.rec, farm=G.rec.str[:3], day=G.day, d=(b1 - y) ** 2 - (b0 - y) ** 2, p2=p2)))
        res.append(dict(val=v, seed=sd_, alone_X=rm(mx - y), alone_EV3L=rm(m0 - y), dSSE_p2=float(((b1 - y) ** 2 - (b0 - y) ** 2)[p2].sum()),
                        blend_vs_sg_X=100 * (rm(b1 - y) / rm(sg - y) - 1), blend_vs_sg_EV3L=100 * (rm(b0 - y) / rm(sg - y) - 1)))
    S = pd.DataFrame(res); B = pd.concat(B, ignore_index=True); print(f"\n=== {kind} vs EV3L ==="); print(S.round(4).to_string(index=False))
    def boot(v, p2only):
        g = B[(B.val == v) & (B.p2 if p2only else True)]; s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
        m = s[np.random.default_rng(20270104).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot("P2LOO", True)
    RD = B[B.val.isin(["P2LOO", "EL1"]) & B.p2].groupby(["val", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum())
    _, pb = boot("DIAG10", False); c3 = not (B[B.val == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
    VER[kind] = c1 and (pP < .0125 or nh == 0) and c3
    print(f"(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed {nh} | (3) DIAG10 share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'} => {kind}: {'ADOPT' if VER[kind] else 'FAIL'}")
print("\nEVX:", VER)
