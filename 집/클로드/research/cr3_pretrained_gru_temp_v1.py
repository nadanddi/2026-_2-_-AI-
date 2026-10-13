# -*- coding: utf-8 -*-
"""CR3 (2026-10-13 집 클로드; 사용자: 'RNN 이 맞는 것 같다 → 다른 49개 온실로 사전학습 해보자').  사전 고정(커밋 후 실행).  온도.
CR1·CR2 실패 원인 = F13/F47 학습 400일·끊긴 시퀀스·추운 날 경험 부족(1.7%).  다른 49개 온실(학습 자료, 사용 제한 없음)은 연속 기록 약 29.5만 행, 추운 행 12.5%.
사전학습(F13/F47 미사용 → 검증 누수 없음): 온실별 연속 날(d−1, d) 48시간 시퀀스.  입력 = 실내온도·실내습도·실내CO₂·시각(sin/cos)
  + 앞 24시간 배지온도 정답(시퀀스의 30% 는 앞날 정답을 가림 → '정답 없음' 표시) + 표시들.  목표 = 뒤 24시간 배지온도.
  모델 GRU(입력→64, 2층, dropout .1) → Linear; Adam lr 2e-3, 8 에폭, 배치 256, 시드 1·2 평균, GPU.  표준화는 사전학습 자료로.
F13/F47 적용: 앞 24시간 = 같은 동 전날(v15R 일치/고확신 → v7 앵커 짝; CR1 과 같음), 그 배지 정답은 학습 기록이고 시험 폴드에서 숨김이 아니면 넣음; 없으면 가림 표시.
안(2개, α .0125):  Z = 사전학습 그대로;  FT = 사전학습 후 F13/F47 학습원(DIAG10 공통, 시험 기록과 같은 온실 ±1·다른 온실 ±5 제외)으로 4 에폭 미세조정(lr 5e-4).
섞기(고정): final = base + w·(GRU − base),  w = .2 + .3·g,  g = clip((10 − in_temp)/2, 0, 1)(결측 0).   base = 14회차형(셀 + v7 앵커 + TNA8 앵커).
판정(온도 범위 ALL): 주 DIAG10·EXT10 모두 4칸 개선 & 농장×5일 블록 부트스트랩 P(악화) < .0125; 나머지 배치는 칸 평균 악화이면서 '후보가 나음' 비율 < .0125 면 FAIL.
서술: GRU 단독 RMSE(배치별·추운 행), 사전학습 자료 검증 RMSE(온실 하나 빼기 아님 — 마지막 10% 날).
"""
import os, sys, importlib.util
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read()
exec(src[:src.index("V = [\"in_temp\", \"in_hum\"")])   # A7, LC, PREV, ST, idx, TRAIN, FARM, NUM, N, rm, DEV
rng = np.random.default_rng(20261124)
VI = ["in_temp", "in_hum", "in_co2"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Yd = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X = X.merge(Yd[["row_id", "sub_temp"]], on="row_id", how="left"); X["farm"] = X.row_id.str[:3]; X["day"] = X.row_id.str[4:7].astype(int); X["h"] = X.row_id.str[8:].astype(int)
O = X[~X.farm.isin(["F13", "F47"])]
def day_cube(D):
    P = {v: D.pivot_table(index=["farm", "day"], columns="h", values=v).reindex(columns=range(24)) for v in VI + ["sub_temp"]}
    return P
P = day_cube(O); P = {v: t.reindex(index=P["in_temp"].index) for v, t in P.items()}; keys = list(P["in_temp"].index); kset = set(keys)
pairs = [(k, (k[0], k[1] - 1)) for k in keys if (k[0], k[1] - 1) in kset]
MU = np.array([np.nanmean(O[v]) for v in VI]); SD = np.array([np.nanstd(O[v]) for v in VI]); SMU, SSD = np.nanmean(O.sub_temp), np.nanstd(O.sub_temp)
HS, HC = np.sin(2 * np.pi * np.arange(24) / 24), np.cos(2 * np.pi * np.arange(24) / 24)
def seq(inp_prev, st_prev, inp_today, show):
    """inp_*: [24,3] raw; st_prev: [24] raw or NaN.  returns [48, 3+2+1+2] standardised"""
    x = np.zeros((48, 8), np.float32)
    x[:24, :3] = (inp_prev - MU) / SD; x[24:, :3] = (inp_today - MU) / SD
    x[:, 3] = np.r_[HS, HS]; x[:, 4] = np.r_[HC, HC]
    if show and np.isfinite(st_prev).any(): x[:24, 5] = (np.nan_to_num(st_prev, nan=SMU) - SMU) / SSD; x[:24, 6] = 1.0
    x[24:, 7] = 1.0
    return np.nan_to_num(x)
def cube_of(k): return np.stack([P[v].loc[k].values for v in VI], 1)
Xs, Ys, last = [], [], []
dmax = {f: max(d for ff, d in keys if ff == f) for f in set(k[0] for k in keys)}
for k, kp in pairs:
    show = rng.random() > .3
    Xs.append(seq(cube_of(kp), P["sub_temp"].loc[kp].values, cube_of(k), show)); Ys.append((P["sub_temp"].loc[k].values - SMU) / SSD)
    last.append(k[1] > dmax[k[0]] * .9)
Xs = np.stack(Xs); Ys = np.stack(Ys).astype(np.float32); last = np.array(last)
print(f"pretraining sequences {len(Xs)} (holdout last-10% days {last.sum()})", flush=True)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.g = nn.GRU(8, 64, num_layers=2, batch_first=True, dropout=.1); s.o = nn.Linear(64, 1)
    def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1)
def train(net, Xa, Ya, epochs, lr, seed):
    torch.manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=lr); xt = torch.tensor(Xa).to(DEV); yt = torch.tensor(np.nan_to_num(Ya)).to(DEV); mt = torch.tensor(np.isfinite(Ya).astype(np.float32)).to(DEV)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(xt), generator=g); net.train()
        for i in range(0, len(xt), 256):
            j = perm[i:i + 256]; out = net(xt[j])[:, 24:]; loss = ((out - yt[j]) ** 2 * mt[j]).sum() / mt[j].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net
def predict(net, Xa):
    with torch.no_grad(): return net(torch.tensor(Xa).to(DEV))[:, 24:].cpu().numpy() * SSD + SMU
nets = []
for seed in (1, 2):
    torch.manual_seed(seed); n = Net().to(DEV); nets.append(train(n, Xs[~last], Ys[~last], 8, 2e-3, seed))
hp = np.mean([predict(n, Xs[last]) for n in nets], 0); yt = Ys[last] * SSD + SMU
print(f"pretrain holdout RMSE {rm(np.nan_to_num(hp - yt)[np.isfinite(yt)]):.3f} (persistence of prev-day mean {rm(np.nan_to_num((Xs[last][:, :24, 5].mean(1, keepdims=True) * SSD + SMU) - yt)[np.isfinite(yt)]):.3f})", flush=True)
nets = []
for seed in (1, 2):
    torch.manual_seed(seed); n = Net().to(DEV); nets.append(train(n, Xs, Ys, 8, 2e-3, seed))
F = X[X.farm.isin(["F13", "F47"])]; T = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); T["farm"] = T.row_id.str[:3]; T["day"] = T.row_id.str[4:7].astype(int); T["h"] = T.row_id.str[8:].astype(int)
FA = pd.concat([F, T]); _ix = pd.MultiIndex.from_frame(FA[["farm", "day"]].drop_duplicates()); CB = {v: FA.pivot_table(index=["farm", "day"], columns="h", values=v).reindex(index=_ix, columns=range(24)) for v in VI}
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
def fk(r): return (r[:3], int(r[4:7]))
def fcube(r): return np.stack([CB[v].loc[fk(r)].values for v in VI], 1)
def fseq(r, hidden):
    pv = PREV.get(r)
    if pv is None: return seq(np.full((24, 3), np.nan), np.full(24, np.nan), fcube(r), False)
    vis = TRAIN[idx[pv]] and pv not in hidden
    return seq(fcube(pv), ST[idx[pv]] if vis else np.full(24, np.nan), fcube(r), vis)
FR = A7.A2.A1.frames()
def layout_maps(M, cells):
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    return key, h, y, anc, recs, pos
_, dh, dy, _, DREC, dpos = layout_maps(*FR["DIAG10"])
DY = {r: np.array([dy[dpos[r]][dh[dpos[r]] == t][0] if (dh[dpos[r]] == t).any() else np.nan for t in range(24)]) for r in DREC}
def purge(te):
    fd = {(FARM[idx[r]], int(NUM[idx[r]])) for r in te}; out = set()
    for r in DREC:
        f, d = FARM[idx[r]], int(NUM[idx[r]])
        if any((f == f2 and abs(d - d2) <= 1) or (f != f2 and abs(d - d2) <= 5) for f2, d2 in fd): out.add(r)
    return out
import copy
res, blocks, solo = [], {}, []
for L, (M, cells) in FR.items():
    key, h, y, anc, recs, pos = layout_maps(M, cells)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; GZ, GF = {}, {}
    for k in sorted(set(fold_of.values())):
        te = [r for r in recs if fold_of[r] == k]; hid = set(te); tr = [r for r in DREC if r not in purge(te)]
        Xte = np.stack([fseq(r, hid) for r in te]); pz = np.mean([predict(n, Xte) for n in nets], 0)
        Xtr = np.stack([fseq(r, hid) for r in tr]); Ytr = np.stack([(DY[r] - SMU) / SSD for r in tr]).astype(np.float32)
        pf = np.mean([predict(train(copy.deepcopy(n), Xtr, Ytr, 4, 5e-4, s + 10), Xte) for s, n in enumerate(nets)], 0)
        for r, a, b in zip(te, pz, pf): GZ[r] = a; GF[r] = b
    gz, gf = np.zeros(len(M)), np.zeros(len(M))
    for r in recs:
        ix = pos[r]; gz[ix] = GZ[r][h[ix]]; gf[ix] = GF[r][h[ix]]
    inT = M.in_temp.values; g = np.nan_to_num(np.clip((10 - inT) / 2, 0, 1), nan=0.0); w = .2 + .3 * g; cold = inT <= 8; ab = np.mean(anc, 0)
    solo.append(dict(layout=L, Z=rm(gz - y), FT=rm(gf - y), base=rm(ab - y), Z_cold=rm((gz - y)[cold]) if cold.any() else np.nan, FT_cold=rm((gf - y)[cold]) if cold.any() else np.nan, base_cold=rm((ab - y)[cold]) if cold.any() else np.nan))
    for ci, base in enumerate(anc):
        for a, gg in (("Z", gz), ("FT", gf)):
            new = base + w * (gg - base); p2 = M.day.values >= 179
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(solo).round(3).to_string(index=False))
Sx = pd.DataFrame(res); print(Sx.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(Sx.groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("Z", "FT"):
    ok = True; msg = []
    for L in sorted(set(Sx.layout)):
        c_ = Sx[(Sx.arm == a) & (Sx.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10"):
            c = bool((c_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((c_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = c_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {c_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== CR3 {a}:"); print("\n".join("   " + m for m in msg)); print(f"CR3 {a}:", "ADOPT" if ok else "FAIL")
