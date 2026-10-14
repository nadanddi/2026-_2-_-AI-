# -*- coding: utf-8 -*-
"""MT4 (2026-10-14 집 클로드; 사용자: '그럼 관수 머리만 붙여봐').  사전 고정(커밋 후 실행).  EC.
MT3 의 BP 는 MT1(남은 하루 입력 보조) + 관수 머리였음 → 여기선 다른 보조 없이 'EC + 관수 머리' 만.
  A  = 일반 GRU(EG1/MT1 의 A: 몸통 → EC 하나), F32 안 섞음                       — 기준
  P  = A 몸통 + 관수 머리(몸통 z → Linear 2: act_pump·act_valve, 가중 .5), F32 학습 기록 233 섞음(F32 는 EC 손실 없음, 관수 손실만)
  몸통·입력·표준화(F13·F47 학습 기록 통계)·25 에폭·배치 16·내부 시드 1·2 = MT3 와 같음.  예측 시 그 기록 자신의 0..h 입력만(5점 유형).
판정(단일 비교 α .025; 오늘 MTL 계열 4번째 시도라는 점은 보고에 명시):  P 가 A 보다 낫다 =
  (a) 단독 RMSE 세 검증기 모두 낮음 AND (b) 섞기(0.9·sg + 0.1·X) dSSE < 0 이 9칸 모두 AND (c) P2LOO 섞기 행 블록 부트스트랩 P(악화) < .025.
서술: B(MT1) 와의 비교, 1차/2차 단독, 2차 보통/고EC.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mt4_ec_plus_irrigation_head_only_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt3_mtl_irrigation_head_f32_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt3_mtl_irrigation_head_f32_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "mt4_ec_plus_irrigation_head_only_v1.py")
class NetP(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, 1); s.p = torch.nn.Linear(128, 2)
    def forward(s, x):
        z = s.g(x)[0]; return s.o(z).squeeze(-1), s.p(z)
def fit4(tr, te, kind):
    it = np.array([idx[r] for r in tr]); ie = np.array([idx[r] for r in te])
    A_ = CUBE2[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    if kind == "P": it = np.concatenate([it, I32])
    def prep(I):
        Z = (CUBE2[I] - mu) / sd; m = np.isnan(CUBE2[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC2[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    Pz = (PUMP2[it] - np.nanmean(P32.reshape(-1, 2), 0)) / (np.nanstd(P32.reshape(-1, 2), 0) + 1e-6)
    pt = torch.tensor(np.nan_to_num(Pz).astype(np.float32), device=DEV); pm = torch.tensor(np.isfinite(Pz).astype(np.float32), device=DEV)
    out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = (NetP(xt.shape[2]) if kind == "P" else NetA(xt.shape[2])).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; mj = mt[j]; o = net(xt[j])
                loss = (((o[0] - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1)
                if kind == "P": loss = loss + .5 * (((o[1] - pt[j]) ** 2) * pm[j]).sum() / pm[j].sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe)[0].cpu().numpy() + ym)
    return np.mean(out, 0)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
M3 = pd.read_pickle(os.path.join(env.LOCAL, "mt3_preds_v1.pkl"))["PRED"]["B"]   # 참고용 B(MT1) — MT3 실행분
KINDS = ("A", "P"); PRED = {k: {} for k in KINDS}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind in KINDS:
            p = fit4(tr, te, kind)
            for j, r in enumerate(te): PRED[kind][(v, r)] = p[j]
    print("done", v, flush=True)
PRED["B"] = M3; res, blk = [], []
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    key = (G.rec.str[:3] + "_" + (G.day // 5).astype(str)).values; BL = {}
    for kind in ("A", "P", "B"):
        m = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m; BL[kind] = bl
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p1=rm((m - y)[~p2]) if (~p2).any() else np.nan, alone_p2=rm((m - y)[p2]),
                        alone_p2_normal=rm((m - y)[p2 & ~hi]), alone_p2_high=rm((m - y)[p2 & hi]), blend_SSE=float(((bl - y) ** 2).sum())))
    if v == "P2LOO": blk.append(pd.Series((BL["P"] - y) ** 2 - (BL["A"] - y) ** 2).groupby(key).mean())
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
a = S[S.kind == "P"].groupby("val").alone.mean() < S[S.kind == "A"].groupby("val").alone.mean(); ca = bool(a.all())
d = S[S.kind == "P"].set_index(["val", "seed"]).blend_SSE - S[S.kind == "A"].set_index(["val", "seed"]).blend_SSE; cb = bool((d < 0).all())
xx = pd.concat(blk, axis=1).mean(axis=1).values; mb = xx[np.random.default_rng(20261226).integers(0, len(xx), (20000, len(xx)))].mean(1); pw = float((mb >= 0).mean())
print(f"P vs A: (a) alone lower all validators {ca} {a.to_dict()} | (b) blend dSSE<0 cells {int((d < 0).sum())}/9 | (c) P2LOO P(worse) {pw:.4f} (< .025) -> {'PASS' if ca and cb and pw < .025 else 'FAIL'}")
db = S[S.kind == "P"].set_index(["val", "seed"]).blend_SSE - S[S.kind == "B"].set_index(["val", "seed"]).blend_SSE
print("descriptive P vs B(MT1): blend dSSE<0 cells", int((db < 0).sum()), "/9 | alone", S[S.kind.isin(["P", "B"])].groupby(["val", "kind"]).alone.mean().unstack().round(4).to_dict())
print("MT4:", "PASS" if ca and cb and pw < .025 else "FAIL")
