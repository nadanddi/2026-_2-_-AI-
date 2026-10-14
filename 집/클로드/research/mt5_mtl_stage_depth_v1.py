# -*- coding: utf-8 -*-
"""MT5 (2026-10-14 집 클로드; 사용자: 'MTL 구조가 3단·4단일 때도 있나?' → '해봐').  사전 고정(커밋 후 실행).  EC.  오늘 MTL 계열 6번째 시도.
MT1(B) = 2단: GRU 몸통 z → 1차 머리 [EC1, AUX1] → mix([z, 1차]) → [EC2, AUX2],  손실 = EC1 + EC2 + .5(AUX1 + AUX2), 출력 EC2.
여기선 단 수만 바꿈(반복 정제 + 단마다 손실 = 중간 감독):
  S1 = 1단(1차 머리만, 손실 EC1 + .5 AUX1)
  S2 = 2단 (= MT1 B 재실행)
  S3 = 3단: 2단 출력을 다시 mix_2([z, 2차]) → [EC3, AUX3]  (단마다 새 mix 층, 64 숨김, 가중 공유 없음)
  S4 = 4단
  손실 = Σ_k (EC_k + .5 AUX_k), 출력 = 마지막 단 EC.  AUX = 남은 하루 입력 7종 평균(MT1 과 같음).  나머지 설정 MT1 과 동일(내부 시드 1·2).
판정(본페로니 k = 2 → .0125):  S3 vs S2, S4 vs S2 각각 — (a) 단독 RMSE 세 검증기 모두 낮음 AND (b) 섞기(0.9·sg + 0.1·X) dSSE < 0 9칸 모두
  AND (c) P2LOO 섞기 행 블록 부트스트랩 P(악화) < .0125.   S1 vs S2 는 서술(MT1 의 2단 구조 자체가 도움이 됐는지).
검증: ec_rv1 CUR 배치(MT1 과 같음).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mt5_mtl_stage_depth_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("def fit_predict")])
__file__ = os.path.join(HERE, "mt5_mtl_stage_depth_v1.py")
class NetS(torch.nn.Module):
    def __init__(s, d, k, n):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.h1 = torch.nn.Linear(128, 1 + k)
        s.mix = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(128 + 1 + k, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1 + k)) for _ in range(n - 1)])
    def forward(s, x):
        z = s.g(x)[0]; o = s.h1(z); outs = [o]
        for m in s.mix: o = m(torch.cat([z, o], -1)); outs.append(o)
        return outs
def fit_s(tr, te, n):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A_ = CUBE[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    F = FUT[it]; fm, fs = np.nanmean(F.reshape(-1, len(AIX)), 0), np.nanstd(F.reshape(-1, len(AIX)), 0) + 1e-6; Fz = (F - fm) / fs
    ft = torch.tensor(np.nan_to_num(Fz).astype(np.float32), device=DEV); fmask = torch.tensor(np.isfinite(Fz).astype(np.float32), device=DEV)
    out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = NetS(xt.shape[2], len(AIX), n).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; mj = mt[j]; fj, fmj = ft[j], fmask[j]; loss = 0
                for o in net(xt[j]):
                    loss = loss + (((o[..., 0] - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1) + .5 * (((o[..., 1:] - fj) ** 2) * fmj).sum() / fmj.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe)[-1][..., 0].cpu().numpy() + ym)
    return np.mean(out, 0)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
KINDS = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}; PRED = {k: {} for k in KINDS}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind, n in KINDS.items():
            p = fit_s(tr, te, n)
            for j, r in enumerate(te): PRED[kind][(v, r)] = p[j]
    print("done", v, flush=True)
res, blk = [], {}
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    key = (G.rec.str[:3] + "_" + (G.day // 5).astype(str)).values; BL = {}
    for kind in KINDS:
        m = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m; BL[kind] = bl
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p1=rm((m - y)[~p2]) if (~p2).any() else np.nan, alone_p2=rm((m - y)[p2]),
                        alone_p2_normal=rm((m - y)[p2 & ~hi]), alone_p2_high=rm((m - y)[p2 & hi]), blend_SSE=float(((bl - y) ** 2).sum())))
    if v == "P2LOO":
        for kind in ("S1", "S3", "S4"): blk.setdefault(kind, []).append(pd.Series((BL[kind] - y) ** 2 - (BL["S2"] - y) ** 2).groupby(key).mean())
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
ok = {}
for kind in ("S1", "S3", "S4"):
    a = S[S.kind == kind].groupby("val").alone.mean() < S[S.kind == "S2"].groupby("val").alone.mean(); ca = bool(a.all())
    d = S[S.kind == kind].set_index(["val", "seed"]).blend_SSE - S[S.kind == "S2"].set_index(["val", "seed"]).blend_SSE; cb = bool((d < 0).all())
    xx = pd.concat(blk[kind], axis=1).mean(axis=1).values; mb = xx[np.random.default_rng(20261226).integers(0, len(xx), (20000, len(xx)))].mean(1); pw = float((mb >= 0).mean())
    tag = "descriptive" if kind == "S1" else ("PASS" if ca and cb and pw < .0125 else "FAIL"); ok[kind] = tag
    print(f"{kind} vs S2: (a) alone lower all validators {ca} {a.to_dict()} | (b) blend dSSE<0 cells {int((d < 0).sum())}/9 | (c) P2LOO P(worse) {pw:.4f} -> {tag}")
print("alone (seed-mean):", S.groupby(["val", "kind"]).alone.mean().unstack().round(4).to_dict())
print("MT5:", ok)
