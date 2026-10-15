# -*- coding: utf-8 -*-
"""CRG (2026-10-15 집 클로드; 사용자: '사슬 GRU 를 병목에만이 아니라 다른 부분도 같이 보면서' → '진행해봐').  사전 고정(커밋 후 실행).  EC.
사슬 잔차 GRU: 같은 동의 앞날 사슬(링커)을 여러 날 읽으며 기본 예측의 잔차를 보정 — 앵커·하루 수준 보정·고EC 에피소드 기억을 한 모델로.
사슬: 패키지 25 의 MASK 세계 거울 링커(v15r_mirror_masked_pkg25.csv) prev0, p0 ≥ .5 를 최대 5단계 거슬러(순환 방지), 학습·검증 모두 같은 규칙
  (CP2 트랙은 정답 연속으로 만든 것이라 검증 기록 정답이 새므로 안 씀).
입력(시간별, 날을 이어 [앞날 k … 앞날 1, 오늘] 최대 6 x 24 = 144 단계, 단방향):
  14 입력 + 시각 sin·cos(EG1 CUBE), base(오늘 위치만; 앞날 NaN), 앞날 실제 EC(정답 있는 학습 기록 & 그 폴드 숨김 아님일 때만; 학습 때 날마다 50% 가림),
  EC 있음 표시, 그 날의 링커 확신(오늘 1), 오늘까지 날 수(0..5)/5, 오늘 표시.  표준화 = 폴드 학습 시퀀스, NaN → 0 + in_temp 결측 표시.
목표: 오늘 위치 잔차 y − base.  학습 기록 base = DIAG10 OOF SG2 전 예측(EC/hr5_train_base_nosg_oof.csv 와 같은 값, ec_rv1 nosg 시드 평균).
모형: GRU(d → 64, 1층) → Linear(1), Adam 1e-3, 40 에폭, 배치 16, 내부 시드 1·2 평균, GPU.
검증: SZ2 체크포인트 62 폴드(NH2·EA3 와 같음), 숨김 = 검증 날 + 같은 농장 ±1 + 다른 농장 ±3 (학습 대상·앞날 EC 채널 모두에서).
후보 C = 0.9·(pre_SEA + .5·CRG) + 0.1·EV3L  vs  F5E = 0.9·(pre_SEA + .5·HR5) + 0.1·EV3L  (HR5 자리를 CRG 로; 반영 .5 는 HR5 와 같게 고정).
판정(EC 분야 배정 규칙, 단일 α .025): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
서술: C 의 S·SH 대비, 사슬 길이별, 앞날 EC 받은 기록 비율, 고EC/보통.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u crg_chain_residual_gru_ec_v1.py
"""
import os, sys, pickle, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("def fit_predict")])
__file__ = os.path.join(HERE, "crg_chain_residual_gru_ec_v1.py")
SEEDS = (1357, 2468, 3579); KMAX = 5; OTH = {"F13": "F47", "F47": "F13"}
L = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); LINK = {r: (p, float(q)) for r, p, q in zip(L.rec, L.prev0, L.p0) if isinstance(p, str) and q >= .5 and p != r and p in idx}
def chain(r):
    out, seen, cur = [], {r}, r
    for _ in range(KMAX):
        l = LINK.get(cur)
        if l is None or l[0] in seen: break
        out.append(l); seen.add(l[0]); cur = l[0]
    return out[::-1]                      # oldest first: [(rec, conf), ...]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy()
NB = O[O.validator == "DIAG10"].groupby("row_id").nosg.mean()
BASE_TR = {}
for r in LABREC:
    b = NB.reindex([f"{r}_{h:02d}" for h in range(24)]).values
    if np.isfinite(b).all(): BASE_TR[r] = b.astype(np.float32)
NC = CUBE.shape[2]; D = NC + 6           # inputs + base, ec, has, conf, k/5, today
def build(r, base_today, vis, drop_rng=None):
    ch = chain(r); days = ch + [(r, 1.0)]; T = len(days) * 24; X = np.full((T, D), np.nan, np.float32); n = len(days)
    for j, (rec, cf) in enumerate(days):
        s = slice(24 * j, 24 * j + 24); X[s, :NC] = CUBE[idx[rec]]; X[s, NC + 3] = cf; X[s, NC + 4] = (n - 1 - j) / KMAX; X[s, NC + 5] = float(rec == r); X[s, NC + 2] = 0.0
        if rec == r: X[s, NC] = base_today
        elif rec in vis and np.isfinite(EC[idx[rec]]).any() and (drop_rng is None or drop_rng.random() >= .5):
            X[s, NC + 1] = EC[idx[rec]]; X[s, NC + 2] = 1.0
    return X
def pad(seqs):
    T = max(len(x) for x in seqs); A = np.full((len(seqs), T, D), np.nan, np.float32)
    for i, x in enumerate(seqs): A[i, T - len(x):] = x          # left pad -> today's 24 steps are always the last 24
    return A
class Net(torch.nn.Module):
    def __init__(s, d):
        super().__init__(); s.g = torch.nn.GRU(d, 64, num_layers=1, batch_first=True); s.o = torch.nn.Linear(64, 1)
    def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1)
def fit_fold(tr_recs, te_items, hide):
    vis = set(LABREC) - hide; ytr = []; rng = np.random.default_rng(0); Xtr = []
    for r in tr_recs:
        y = EC[idx[r]] - BASE_TR[r]
        for rep in range(2): Xtr.append(build(r, BASE_TR[r], vis, rng)); ytr.append(y)
    A = pad(Xtr); mu = np.nanmean(A.reshape(-1, D), 0); sd = np.nanstd(A.reshape(-1, D), 0) + 1e-6
    def prep(B):
        Z = (B - mu) / sd; m = np.isnan(B[..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt = prep(A); Y = np.stack(ytr).astype(np.float32); yt = torch.tensor(np.nan_to_num(Y), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    Xte = prep(pad([build(r, b, vis) for r, b in te_items])); out = []
    for seed in (1, 2):
        torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(40):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; e = net(xt[j])[:, -24:]; mj = mt[j]; loss = (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(Xte)[:, -24:].cpu().numpy())
    return np.mean(out, 0)
CKS = os.path.join(env.LOCAL, "sz2_ckpt"); CF = os.path.join(env.LOCAL, "crg_preds_v1.pkl"); CP = pickle.load(open(CF, "rb")) if os.path.exists(CF) else {}
for f in sorted(glob.glob(os.path.join(CKS, "*.csv"))):
    G = pd.read_csv(f); v, k = G.validator.iloc[0], int(G.validation_fold.iloc[0])
    if (v, k) in CP: continue
    G["rec"] = G.farm + "_" + G.day.map("{:03d}".format); vd = {(r[:3], int(r[4:])) for r in G.rec.unique()}
    hide = {r for r in LABREC if any((r[:3] == ff and abs(int(r[4:]) - dd) <= 1) or (r[:3] == OTH[ff] and abs(int(r[4:]) - dd) <= 3) for ff, dd in vd)} | set(G.rec)
    tr = [r for r in BASE_TR if r not in hide]; items, keys = [], []
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour")
        if len(g) != 24 or r not in idx: continue
        for s in SEEDS: items.append((r, g[f"pre_SEA_{s}"].values.astype(np.float32))); keys.append((r, s))
    p = fit_fold(tr, items, hide); CP[(v, k)] = {key: p[j] for j, key in enumerate(keys)}
    pickle.dump(CP, open(CF, "wb")); print(v, k, "done | chain len mean %.2f" % np.mean([len(chain(r)) for r, _ in items]), flush=True)
# ---- judge ----
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MTp = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
Z = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CKS, "*.csv"))], ignore_index=True).rename(columns={"validation_fold": "fold"}); Z["rec"] = Z.farm + "_" + Z.day.map("{:03d}".format)
R = R.merge(Z[["validator", "fold", "rec", "hour"] + [f"pre_SEA_{s}" for s in SEEDS]], on=["validator", "fold", "rec", "hour"], how="left")
mt = np.array([MTp[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
R["clen"] = [len(chain(r)) for r in R.rec]
for s in SEEDS:
    cr = np.array([CP[(v, k)][(r, s)][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
    R[f"F5E_{s}"] = .9 * ((R[f"A1_{s}"] - .1 * mt) / .9) + .1 * ev; R[f"C_{s}"] = .9 * (R[f"pre_SEA_{s}"] + .5 * cr) + .1 * ev
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
rows, blocks, recd = [], {}, []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; hi = g.hi.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; led = ~g.rec.isin(LEDGER).values
    for s in SEEDS:
        a, n = g[f"F5E_{s}"].values, g[f"C_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
        fx = lambda ref, m: 100 * (rm((n - y)[m]) / rm((g[ref].values - y)[m]) - 1) if m.any() else np.nan
        rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), vsF5E_all=fx(f"F5E_{s}", np.ones(len(y), bool)), vsF5E_p2=fx(f"F5E_{s}", p2),
                         p2_normal=fx(f"F5E_{s}", p2 & ~hi), p2_high=fx(f"F5E_{s}", p2 & hi), vsS_p2=fx(f"S_{s}", p2), vsSH_p2=fx(f"SH_{s}", p2), vsSH_all=fx(f"SH_{s}", np.ones(len(y), bool)), X0_p2=float(d[p2 & led].sum())))
        blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
        for r_, dd in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd))
S = pd.DataFrame(rows); pd.set_option("display.width", 240); print(S.round(3).to_string(index=False))
def boot(key_):
    x = pd.concat(blocks[key_], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20270105).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot(("P2LOO", "p2"))
RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum())
dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
print(f"\n(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("by chain length (P2LOO+EL1 pass-2, seed-mean dSSE):", R[R.validator.isin(["P2LOO", "EL1"]) & R.p2].assign(d=np.mean([(R[f"C_{s}"] - R.y) ** 2 - (R[f"F5E_{s}"] - R.y) ** 2 for s in SEEDS], 0)[R.validator.isin(["P2LOO", "EL1"]) & R.p2]).groupby("clen").d.sum().round(3).to_dict())
print("P2LOO worst/best records:", RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values().round(3).head(4).to_dict(), RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values().round(3).tail(4).to_dict())
print("CRG:", "ADOPT" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
