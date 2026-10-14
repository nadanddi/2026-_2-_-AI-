# -*- coding: utf-8 -*-
"""MT1 (2026-10-14 집 클로드; 사용자: 박사 형 구조 'input > (EC, 입력변수1, 입력변수2) > 상호작용 층 > (EC, 입력변수1, 입력변수2)', 한 신경망 안).  사전 고정(커밋 후 실행).  EC.
가설: 보조 과제로 '그날 남은 시각의 입력'을 같이 예측하게 하면(학습 기록의 미래 입력은 학습 목표로만; 예측 때는 0..h 입력만 → 인과),
  모델이 앞으로의 운영을 내다본 표현을 배워 EC 가 나아짐 (HR2: 하루 전체 입력을 알면 하루 수준 −11%).
A (기준) = EG1 과 같은 하루 GRU(입력 14 + 시각 sin/cos + in_temp 결측 표시, 2층 128) → 시각별 EC.
B (MTL-상호작용) = 같은 몸통 → 1차 머리: EC1_h, AUX1_h(7) → 상호작용 층: [몸통_h, EC1_h, AUX1_h] → MLP(64) → EC2_h, AUX2_h.
  AUX 목표 = 시각 h 이후(h+1..23) 그날 평균: in_temp, in_hum, in_co2, act_thermal, act_vent, act_co2, out_rad (학습 폴드로 표준화, h = 23 은 제외).
  손실 = EC1 + EC2 + 0.5 (AUX1 + AUX2), 최종 EC = EC2.
공통: 목표 EC − 학습 평균, Adam 2e-3, 25 에폭, 배치 16 기록, 시드 1·2 평균, GPU.  폴드·숨김 = EG1 과 같음(DIAG10/P2LOO/EL1, ec_rv1 CUR, EL1 은 이웃 ±1 숨김).
비교(미리 정함): B 가 A 보다 낫다 = 세 검증기 모두 단독 RMSE(전체 행) 낮음 AND 0.9 sg + 0.1 모델 섞기 ΔSSE 가 9칸 모두 A 보다 작음.
서술: 단독/섞기 RMSE, 2차·보통/고EC, AUX 예측 정확도(상관).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eg1_ec_dayseq_gru_member_v1.py")
exec(src[:src.index("class Net")])
__file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
AUXV = ["in_temp", "in_hum", "in_co2", "act_thermal", "act_vent", "act_co2", "out_rad"]; AIX = [USE.index(v) for v in AUXV]
RAW = CUBE[:, :, :len(USE)]
with np.errstate(all="ignore"):
    FUT = np.stack([np.nanmean(RAW[:, h + 1:, :][:, :, AIX], 1) if h < 23 else np.full((N, len(AIX)), np.nan) for h in range(24)], 1)   # [N, 24, 7]
class NetA(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, 1)
    def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1), None, None, None
class NetB(torch.nn.Module):
    def __init__(s, d, k):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.h1 = torch.nn.Linear(128, 1 + k)
        s.mix = torch.nn.Sequential(torch.nn.Linear(128 + 1 + k, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1 + k))
    def forward(s, x):
        z = s.g(x)[0]; o1 = s.h1(z); o2 = s.mix(torch.cat([z, o1], -1))
        return o2[..., 0], o1[..., 0], o1[..., 1:], o2[..., 1:]
def fit_predict(tr, te, kind):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A_ = CUBE[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    F = FUT[it]; fm, fs = np.nanmean(F.reshape(-1, len(AIX)), 0), np.nanstd(F.reshape(-1, len(AIX)), 0) + 1e-6; Fz = (F - fm) / fs
    ft = torch.tensor(np.nan_to_num(Fz).astype(np.float32), device=DEV); fmask = torch.tensor(np.isfinite(Fz).astype(np.float32), device=DEV)
    out, aux_out = [], []
    for seed in (1, 2):
        torch.manual_seed(seed); net = (NetA(xt.shape[2]) if kind == "A" else NetB(xt.shape[2], len(AIX))).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; e2, e1, a1, a2 = net(xt[j]); mj = mt[j]
                loss = (((e2 - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1)
                if kind == "B":
                    fj, fmj = ft[j], fmask[j]
                    loss = loss + (((e1 - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1) + .5 * ((((a1 - fj) ** 2) * fmj).sum() + (((a2 - fj) ** 2) * fmj).sum()) / fmj.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            e2, _, _, a2 = net(xe); out.append(e2.cpu().numpy() + ym)
            if kind == "B": aux_out.append(a2.cpu().numpy() * fs + fm)
    return np.mean(out, 0), (np.mean(aux_out, 0) if aux_out else None)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
PRED = {"A": {}, "B": {}}; AUXP = {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for k, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind in ("A", "B"):
            p, a = fit_predict(tr, te, kind)
            for j, r in enumerate(te):
                PRED[kind][(v, r)] = p[j]
                if a is not None: AUXP[(v, r)] = a[j]
    print("done", v, flush=True)
rows = []; res = []
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    for kind in ("A", "B"):
        m = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p2=rm((m - y)[p2]), blend_dSSE=float(((bl - y) ** 2 - (sg - y) ** 2).sum()),
                        blend_pct=100 * (rm(bl - y) / rm(sg - y) - 1), blend_p2_pct=100 * (rm((bl - y)[p2]) / rm((sg - y)[p2]) - 1),
                        blend_p2_normal=100 * (rm((bl - y)[p2 & ~hi]) / rm((sg - y)[p2 & ~hi]) - 1), blend_p2_high=100 * (rm((bl - y)[p2 & hi]) / rm((sg - y)[p2 & hi]) - 1)))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
W = S.pivot_table(index=["val", "seed"], columns="kind", values=["alone", "blend_dSSE"])
c1 = bool((S[S.kind == "B"].groupby("val").alone.mean() < S[S.kind == "A"].groupby("val").alone.mean()).all()); c2 = bool((W[("blend_dSSE", "B")] < W[("blend_dSSE", "A")]).all())
print(f"\nB alone lower in all validators: {c1} | B blend dSSE < A in all 9 cells: {c2} ({int((W[('blend_dSSE', 'B')] < W[('blend_dSSE', 'A')]).sum())}/9)")
cs = []
for (v, r), a in AUXP.items():
    t = FUT[idx[r]][:23]; aa = a[:23]
    for j in range(len(AIX)):
        if np.isfinite(t[:, j]).sum() > 5: cs.append(dict(var=AUXV[j], c=np.corrcoef(aa[np.isfinite(t[:, j]), j], t[np.isfinite(t[:, j]), j])[0, 1]))
print("AUX forecast corr (within-record over hours, median):", pd.DataFrame(cs).groupby("var").c.median().round(3).to_dict())
print("MT1:", "MTL better" if (c1 and c2) else "MTL not better")
