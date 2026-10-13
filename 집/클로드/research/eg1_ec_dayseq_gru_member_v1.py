# -*- coding: utf-8 -*-
"""EG1 (2026-10-14 집 클로드; 사용자: 'RNN 을 조금 손보는 건? → EC 하루 시퀀스 GRU 진행').  사전 고정(커밋 후 실행).  EC.
근거: EC 에 시퀀스 모델은 처음.  고EC 동을 알아보는 단서가 하루 안 '운영 일정 지문'(커튼 9→16시, CO₂ 9·11·13시, 유동팬 꺼짐; HS2 6.690) —
  시간 순서 패턴이라 GRU 가 나무 모델의 누적 요약보다 잘 잡을 수 있다는 가설.  온도 2층 GRU(V3, 6.531·6.532)를 EC 로.
입력: 한 기록(하루) 안 0..h 시각의 입력 14종(out_temp, out_hum, out_rad, out_wspd, in_temp, in_hum, in_co2, act_vent, act_shade, act_thermal,
  act_heating, act_circfan, act_co2, act_fog) + 시각 sin/cos + 결측 표시 1; 자기 기록 입력만(다른 기록·평가 입력 없음 → 규정 안).
  표준화는 그 폴드 학습 기록으로.  단방향 GRU 라 h 시 출력은 0..h 만 봄.
모델: GRU 2층·숨김 128 → Linear(1), 목표 = 그 시각 sub_ec(학습 평균 뺌), 24 출력 모두 손실(정답 있는 시각만), Adam 2e-3, 25 에폭, 배치 16 기록, 시드 1·2 평균, GPU.
학습 기록: 정답 있는 학습 기록 중 그 폴드 숨김 제외 — DIAG10 = 폴드 기록, P2LOO = 그 기록, EL1 = 그 5기록 묶음 + 같은 농장 ±1 번호(6.166 배치).
후보: 0.9 · sg + 0.1 · GRU (sg = ec_rv1 CUR, 시드 47/1414/6464; 섞는 비율 미리 고정).
판정(EC 분야 배정 규칙, 링커 안 씀 → EV 해당 없음): (1) 주 P2LOO·EL1 모든 시드 ΔSSE < 0 AND (P2LOO 시드 평균 농장×5일 블록 P(악화) < .025 OR 나빠진 기록 0);
  (3) DIAG10 손해 감시: 시드 평균 악화 AND '후보 나음' 비율 < .025 이면 FAIL.
서술: GRU 단독 RMSE, sg 와 오차 상관, 2차 고EC/보통 분해, 가짜 고EC(F13_233·F47_161 등) 변화.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); DEV = "cuda" if torch.cuda.is_available() else "cpu"
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CUBE = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in USE], 2).astype(np.float32)
hh = np.arange(24); CUBE = np.concatenate([CUBE, np.broadcast_to(np.stack([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)], 1), (N, 24, 2)).astype(np.float32)], 2)
LABREC = [rid(i) for i in range(N) if TRAIN[i] and np.isfinite(DM[i])]
class Net(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, 1)
    def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1)
def fit_predict(tr, te):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A = CUBE[it]; mu = np.nanmean(A.reshape(-1, A.shape[2]), 0); sd = np.nanstd(A.reshape(-1, A.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV); out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; loss = (((net(xt[j]) - yt[j]) ** 2) * mt[j]).sum() / mt[j].sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe).cpu().numpy() + ym)
    return np.mean(out, 0)   # [len(te), 24]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
GRU = {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]; P = {}
    for k, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set())
        tr = [r for r in LABREC if r not in hide]; pr = fit_predict(tr, te)
        for r, p in zip(te, pr): P[r] = p
    GRU[v] = P; print("done", v, len(P), flush=True)
res, blocks, recd = [], {}, []
for (v, sd), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values
    gr = np.array([GRU[v][r][h] for r, h in zip(G.rec, G.hour)]); new = .9 * sg + .1 * gr
    p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    f = lambda a, m: 100 * (rm((a - y)[m]) / rm((sg - y)[m]) - 1) if m.any() else np.nan
    d = (new - y) ** 2 - (sg - y) ** 2
    res.append(dict(val=v, seed=sd, dSSE=float(d.sum()), all=f(new, np.ones(len(y), bool)), p2=f(new, p2), p2_high=f(new, p2 & hi), p2_norm=f(new, p2 & ~hi),
                    gru_alone=f(gr, np.ones(len(y), bool)), err_corr=float(np.corrcoef(gr - y, sg - y)[0, 1])))
    blocks.setdefault(v, []).append(pd.Series(d).groupby((G.farm + "_" + (G.day // 5).astype(str)).values).mean())
    for r, dd in pd.Series(d).groupby(G.rec.values).sum().items(): recd.append(dict(val=v, seed=sd, rec=r, dsse=dd))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
RD = pd.DataFrame(recd)
print("\nfalse-high / key records (seed-mean dSSE):", RD[RD.rec.isin(["F13_233", "F47_161", "F13_231", "F47_231", "F13_243", "F13_217", "F47_229"])].groupby(["val", "rec"]).dsse.mean().round(3).to_dict())
def boot(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = np.random.default_rng(20261202).integers(0, len(s), (20000, len(s))); m = s[ix].mean(1)
    return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE < 0).all()); pP2, _ = boot("P2LOO")
harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum())
c1b = pP2 < .025 or nh == 0
print(f"\n(1) main P2LOO+EL1: all cells better {c1} ({int((main.dSSE < 0).sum())}/{len(main)}) | P2LOO P(worse) {pP2:.4f} | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot("DIAG10"); c3 = not (dg.dSSE.mean() > 0 and pb < .025)
print(f"(3) harm guard DIAG10: mean dSSE {dg.dSSE.mean():+.3f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("EG1 EC day-sequence GRU member (specialty rule):", "ADOPT" if (c1 and c1b and c3) else "FAIL")
pd.DataFrame([dict(val=v, rec=r, **{f"h{h}": p[h] for h in range(24)}) for v, P in GRU.items() for r, p in P.items()]).to_csv(os.path.join(HERE, "eg1_gru_preds_v1.csv"), index=False)
