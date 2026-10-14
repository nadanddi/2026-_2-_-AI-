# -*- coding: utf-8 -*-
"""CB1 (2026-10-14 집 클로드; 사용자: 'MTL 이것저것 합쳐볼 만하지 않아? → 1번(HR5 + MTL 미래 입력 예측) 진행').  사전 고정(커밋 후 실행).  EC.  HR5 계열 2번째 시도.
근거: HR2(비인과, 하루 전체 입력 요약) 전체 −11% vs HR5(인과, 0..h' 요약) — 차이 = 남은 시각 입력 정보.  MT1(6.894): GRU 가 그날 남은 시각 입력을 정확히 예측(커튼·일사 .98~.99, 실내온 .94).
설계: HR5 와 똑같이(기준 시각 0·3·…·21 TabPFN v2, 목표 = 하루 수준 잔차, 학습 = cp2 360, 숨김 = 검증 + 같은 농장 ±1 + 다른 농장 ±3, 기준 = ec_rv1 CUR sg, λ = .5 고정(HR5 선택값))
  특징 45 + 7: 7 변수(in_temp, in_hum, in_co2, act_thermal, act_vent, act_co2, out_rad)의 '하루 전체 평균 추정' = ((h'+1)·관측 0..h' 평균 + (23−h')·예측 남은 시각 평균)/24.
  미래 입력 예측기 = MT1 몸통(2층 GRU 128, 입력 14 + 시각 sin/cos + 결측 표시)에 보조 머리만(EC 머리 없음 — EC 정답이 예측기를 통해 새는 길 차단), 손실 = 남은 시각 7변수 평균(학습 폴드 표준화),
    학습 기록 400(정답 있는 학습 기록 입력만, 평가 입력 안 씀) 을 농장 x 번호 20 묶음으로 10폴드 → 모든 학습 기록에 OOF 예측(자기 기록 미사용), 시드 1·2 평균, 25 에폭.
판정(미리 정함): (1) HR5 와 같은 EC 규칙 — P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .025 or 손해 0) & DIAG10 손해 감시;
  (2) 개선 인정 = (1) 통과 AND 9칸(검증기 3 x 시드 3) 모두 2차 dSSE 가 HR5(hr5_rows_v1.csv.gz) 보다 작음.
서술: 장부 8기록 제외 dSSE(사용자 미결정 X0 — 서술만), 보통/고EC, 예측기 OOF 정확도.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py")
exec(src[:src.index("TRK = sorted(TR)")])
__file__ = os.path.join(HERE, "cb1_hr5_plus_future_input_forecast_v1.py")
AUXV = ["in_temp", "in_hum", "in_co2", "act_thermal", "act_vent", "act_co2", "out_rad"]; AIX = [USE.index(v) for v in AUXV]
# ---------- future-input forecaster (inputs only) ----------
LABR = sorted(r for r in INP if r[:3] in ("F13", "F47") and r in set(A.rec) | set(TR))   # labelled training records with inputs
Xtr = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TRAINREC = sorted(set(Xtr.row_id.str[:7]) & {r for r in INP if r[:3] in ("F13", "F47")})
hh = np.arange(24); SC = np.stack([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)], 1).astype(np.float32)
def seq(r): return np.concatenate([INP[r].astype(np.float32), SC], 1)
def fut(r):
    R = INP[r][:, AIX]
    with np.errstate(all="ignore"): return np.stack([np.nanmean(R[h + 1:], 0) if h < 23 else np.full(len(AIX), np.nan) for h in range(24)])
class FNet(torch.nn.Module):
    def __init__(s, d, k):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.o = torch.nn.Linear(128, k)
    def forward(s, x): return s.o(s.g(x)[0])
def fit_forecaster(tr, te):
    S = np.stack([seq(r) for r in tr]); mu = np.nanmean(S.reshape(-1, S.shape[2]), 0); sd = np.nanstd(S.reshape(-1, S.shape[2]), 0) + 1e-6
    def prep(rs):
        Z = (np.stack([seq(r) for r in rs]) - mu) / sd; m = np.isnan(Z[..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    F = np.stack([fut(r) for r in tr]); fm, fs = np.nanmean(F.reshape(-1, len(AIX)), 0), np.nanstd(F.reshape(-1, len(AIX)), 0) + 1e-6; Fz = (F - fm) / fs
    xt, xe = prep(tr), prep(te); ft = torch.tensor(np.nan_to_num(Fz).astype(np.float32), device=DEV); fk = torch.tensor(np.isfinite(Fz).astype(np.float32), device=DEV); out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = FNet(xt.shape[2], len(AIX)).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for _ in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; loss = (((net(xt[j]) - ft[j]) ** 2) * fk[j]).sum() / fk[j].sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe).cpu().numpy() * fs + fm)
    return np.mean(out, 0)   # [len(te), 24, 7]
grp = {r: f"{r[:3]}_{int(r[4:]) // 20}" for r in TRAINREC}; G = sorted(set(grp.values())); fold_of_g = {g_: i % 10 for i, g_ in enumerate(G)}
FC = {}
for k in range(10):
    te = [r for r in TRAINREC if fold_of_g[grp[r]] == k]; tr = [r for r in TRAINREC if fold_of_g[grp[r]] != k]
    if te:
        P_ = fit_forecaster(tr, te)
        for r, p in zip(te, P_): FC[r] = p
print("forecaster OOF done:", len(FC), "records", flush=True)
cs = []
for r, p in FC.items():
    t = fut(r)
    for j in range(len(AIX)):
        m = np.isfinite(t[:23, j]) & np.isfinite(p[:23, j])
        if m.sum() > 5 and np.nanstd(t[:23, j][m]) > 1e-9: cs.append(dict(var=AUXV[j], c=np.corrcoef(p[:23, j][m], t[:23, j][m])[0, 1]))
print("forecaster OOF within-record corr (median):", pd.DataFrame(cs).groupby("var").c.median().round(3).to_dict(), flush=True)
def feat2(r, base, h):
    f = feat(INP[r], base, h)
    with np.errstate(all="ignore"): obs = np.nanmean(INP[r][:h + 1, AIX], 0)
    est = ((h + 1) * obs + (23 - h) * FC[r][h]) / 24 if h < 23 else obs
    return np.concatenate([f, est])
# ---------- HR5 harness with feat2 ----------
TR = {}
for r, g in A.groupby("rec"):
    if len(g) == 24 and np.isfinite(g.base_oof.values).all() and np.isfinite(g.y.values).sum() >= 12 and r in INP and r in FC:
        TR[r] = (g.base_oof.values, float(np.nanmean(g.y.values) - g.base_oof.mean()))
TRK = sorted(TR); TF = {h: np.stack([feat2(r, TR[r][0], h) for r in TRK]) for h in HS}; TD = np.array([TR[r][1] for r in TRK])
for h in HS: TF[h] = np.where(np.isfinite(TF[h]), TF[h], np.nanmean(TF[h], 0))
tfarm = np.array([r[:3] for r in TRK]); tnum = np.array([int(r[4:]) for r in TRK]); print("corrector training records", len(TRK), flush=True)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
DH = {}
for v in ("DIAG10", "EL1", "P2LOO"):
    for k, Gk in O[O.validator == v].groupby("fold"):
        val = set(Gk.rec); hide = np.zeros(len(TRK), bool)
        for r in val:
            f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTHER[f]) & (np.abs(tnum - n) <= 3))
        models = {}
        for h in HS:
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True); m.fit(TF[h][~hide], TD[~hide]); models[h] = m
        for (sd, r), g in Gk.groupby(["seed", "rec"]):
            g = g.sort_values("hour"); base = g.sg.values
            if len(base) != 24 or r not in FC: continue
            Q = np.stack([feat2(r, base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS]))
            dh = {h: float(models[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}
            DH[(v, k, sd, r)] = np.array([dh[max(x for x in HS if x <= hh_)] for hh_ in range(24)])
        print(v, k, "done", flush=True)
rows = []
for (v, k, sd), G_ in O.groupby(["validator", "fold", "seed"]):
    for r, g in G_.groupby("rec"):
        g = g.sort_values("hour")
        if (v, k, sd, r) in DH: rows.append(pd.DataFrame(dict(validator=v, seed=sd, rec=r, farm=g.farm.values, day=g.day.values, hour=g.hour.values, y=g.sub_ec.values, base=g.sg.values, new=g.sg.values + .5 * DH[(v, k, sd, r)])))
B = pd.concat(rows, ignore_index=True); B["d"] = (B.new - B.y) ** 2 - (B.base - B.y) ** 2; B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
H5 = pd.read_csv(os.path.join(HERE, "hr5_rows_v1.csv.gz")); H5["d"] = (H5.new - H5.y) ** 2 - (H5.base - H5.y) ** 2
res = []
for (v, sd), g in B.groupby(["validator", "seed"]):
    p2 = g.p2.values; f = lambda m: 100 * (rm((g.new - g.y)[m]) / rm((g.base - g.y)[m]) - 1)
    h5 = H5[(H5.validator == v) & (H5.seed == sd) & (H5.day >= 179)]
    res.append(dict(val=v, seed=sd, dSSE_p2=float(g.d[p2].sum()), hr5_dSSE_p2=float(h5.d.sum()), p2=f(p2), all=f(np.ones(len(g), bool)), p2_normal=f(p2 & ~g.hi.values), p2_high=f(p2 & g.hi.values),
                    p2_noledger=float(g.d[p2 & ~g.rec.isin(LEDGER).values].sum()), hr5_noledger=float(h5[~h5.rec.isin(LEDGER)].d.sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 240); print(S.round(3).to_string(index=False))
def boot(v, p2only):
    g = B[(B.validator == v) & (B.p2 if p2only else True)]
    s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20261225).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot("P2LOO", True)
RD = B[B.validator.isin(["P2LOO", "EL1"]) & B.p2].groupby(["validator", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum()); c1b = pP2 < .025 or nh == 0
_, pb = boot("DIAG10", False); c3 = not (B[B.validator == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
beat = bool((S.dSSE_p2 < S.hr5_dSSE_p2).all())
print(f"\n(1) rule: main cells {int((main.dSSE_p2 < 0).sum())}/{len(main)} | P2LOO P {pP2:.4f} | harmed {nh} | DIAG10 guard {'PASS' if c3 else 'FAIL'} -> {'PASS' if c1 and c1b and c3 else 'FAIL'}")
print(f"(2) better than HR5 in all 9 cells: {beat} ({int((S.dSSE_p2 < S.hr5_dSSE_p2).sum())}/9)")
print("CB1:", "ADOPT (improves HR5)" if (c1 and c1b and c3 and beat) else ("rule PASS but not better than HR5" if (c1 and c1b and c3) else "FAIL"))
B.to_csv(os.path.join(HERE, "cb1_rows_v1.csv.gz"), index=False)
