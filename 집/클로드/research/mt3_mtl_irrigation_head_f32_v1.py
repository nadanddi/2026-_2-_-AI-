# -*- coding: utf-8 -*-
"""MT3 (2026-10-14 집 클로드; 사용자: 박사 형 MTL 은 '배지온도, 배지 EC, 관수 이력' 을 함께 맞힘 → 'mt3 도 해봐').  사전 고정(커밋 후 실행).  EC.
관수 이력(act_pump 양액 펌프 가동률, act_valve)은 F13·F47 학습·평가 모두 100% 결측(3.7·6.114); 51 온실 중 F32 만 있음(98%).
→ F32 학습 기록(233, EC 정답 없음, sub_temp 있음)을 몸통 학습에 섞고, 관수 머리(펌프·밸브 2개)는 F32 행에서만 학습.
  F13·F47 에서는 환경 입력만으로 '관수 중인가' 를 추정하는 표현이 몸통에 생기길 기대(카탈로그 6.545: F32 펌프 지문을 '특징' 으로 옮긴 건
  계절 빼면 EC 상관 ≈ 0 → 기대 낮음; 몸통 공동 학습은 처음).  학습 자료 자유 사용 규칙(train-data-unrestricted) 안.
  F32 는 out_hum·act_shade·act_thermal·act_heating·act_fog 가 없음 → 표준화 후 0 으로 채움(MT1 과 같은 처리).  F32 EC 손실 없음.
안 (MT2 와 같은 몸통·학습 설정·내부 시드 1·2; 표준화 통계는 F13·F47 학습 기록만 → 짝끼리 입력 척도 동일):
  B  = MT1(보조 = 남은 하루 입력 7종)                 BP = B + F32 섞기 + 관수 머리(몸통 z → Linear 2, 가중 .5)
  T  = MT2 의 T(보조 = 배지온도, 배지온도 − 실내)      TP = T + F32 섞기 + 관수 머리
  F32 행의 보조 과제(B: 남은 하루 입력, T: 배지온도)도 값이 있으면 학습에 씀.
판정(짝마다, 본페로니 k = 2 → .0125): X = BP (vs B), TP (vs T).  X 가 짝보다 낫다 =
  (a) 단독 RMSE 세 검증기 모두 낮음 AND (b) 섞기(0.9·sg + 0.1·X) dSSE < 0 이 9칸 모두 AND (c) P2LOO 섞기 행 블록 부트스트랩 P(악화) < .0125.
검증: ec_rv1 CUR 배치(MT1·MT2 와 같음).  서술: 관수 머리의 F32 OOF 상관(F32 기록 5분할 한 번), F13·F47 추정 펌프 가동률과 EC 상관.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mt3_mtl_irrigation_head_f32_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "mt3_mtl_irrigation_head_f32_v1.py")
_X3 = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); _X3 = _X3[_X3.row_id.str[:3] == "F32"]; _X3["rec"] = _X3.row_id.str[:7]; _X3["h"] = _X3.row_id.str[8:].astype(int)
R32 = sorted(_X3.rec.unique()); M = len(R32)
C32 = np.stack([_X3.pivot(index="rec", columns="h", values=v).reindex(index=R32, columns=range(24)).values for v in USE], 2).astype(np.float32)
C32 = np.concatenate([C32, np.broadcast_to(np.stack([np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24)], 1), (M, 24, 2)).astype(np.float32)], 2)
P32 = np.stack([_X3.pivot(index="rec", columns="h", values=v).reindex(index=R32, columns=range(24)).values for v in ("act_pump", "act_valve")], 2).astype(np.float32)
_Yall = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); _Yall = _Yall[_Yall.row_id.str[:3] == "F32"]; _Yall["rec"] = _Yall.row_id.str[:7]; _Yall["h"] = _Yall.row_id.str[8:].astype(int)
T32 = _Yall.pivot(index="rec", columns="h", values="sub_temp").reindex(index=R32, columns=range(24)).values.astype(np.float32)
RAW32 = C32[:, :, :len(USE)]
with np.errstate(all="ignore"):
    FUT32 = np.stack([np.nanmean(RAW32[:, h + 1:, :][:, :, AIX], 1) if h < 23 else np.full((M, len(AIX)), np.nan) for h in range(24)], 1)
CUBE2 = np.concatenate([CUBE, C32]); EC2 = np.concatenate([EC, np.full((M, 24), np.nan, np.float32)]); FUT2 = np.concatenate([FUT, FUT32])
TAUX2 = np.concatenate([TAUX, np.stack([T32, T32 - RAW32[:, :, USE.index("in_temp")]], 2)])
PUMP2 = np.concatenate([np.full((N, 24, 2), np.nan, np.float32), P32]); I32 = np.arange(N, N + M)
print("F32 records", M, "pump non-null", float(np.isfinite(P32).mean()), flush=True)
class Wrap(torch.nn.Module):
    def __init__(s, d, k):
        super().__init__(); s.b = NetB(d, k); s.p = torch.nn.Linear(128, 2)
    def forward(s, x):
        z = s.b.g(x)[0]; o1 = s.b.h1(z); o2 = s.b.mix(torch.cat([z, o1], -1))
        return o2[..., 0], o1[..., 0], o1[..., 1:], o2[..., 1:], s.p(z)
def fit_predict3(tr, te, kind, f32_idx=None):
    pump = kind.endswith("P"); aux = FUT2 if kind[0] == "B" else TAUX2
    it = np.array([idx[r] for r in tr]); ie = np.array([idx[r] for r in te]) if f32_idx is None else f32_idx[1]
    A_ = CUBE2[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    if pump: it = np.concatenate([it, I32 if f32_idx is None else f32_idx[0]])
    def prep(I):
        Z = (CUBE2[I] - mu) / sd; m = np.isnan(CUBE2[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC2[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    F = aux[it]; k = F.shape[2]; fm, fs = np.nanmean(F.reshape(-1, k), 0), np.nanstd(F.reshape(-1, k), 0) + 1e-6; Fz = (F - fm) / fs
    ft = torch.tensor(np.nan_to_num(Fz).astype(np.float32), device=DEV); fmask = torch.tensor(np.isfinite(Fz).astype(np.float32), device=DEV)
    P = PUMP2[it]; pmu, psd = np.nanmean(P32.reshape(-1, 2), 0), np.nanstd(P32.reshape(-1, 2), 0) + 1e-6; Pz = (P - pmu) / psd
    pt = torch.tensor(np.nan_to_num(Pz).astype(np.float32), device=DEV); pmask = torch.tensor(np.isfinite(Pz).astype(np.float32), device=DEV)
    out, pout = [], []
    for seed in (1, 2):
        torch.manual_seed(seed); net = (Wrap(xt.shape[2], k) if pump else NetB(xt.shape[2], k)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; mj = mt[j]; fj, fmj = ft[j], fmask[j]
                ec_l = lambda e: (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1); aux_l = lambda a: (((a - fj) ** 2) * fmj).sum() / fmj.sum().clamp(min=1)
                o = net(xt[j]); loss = ec_l(o[0]) + ec_l(o[1]) + .5 * (aux_l(o[2]) + aux_l(o[3]))
                if pump: loss = loss + .5 * (((o[4] - pt[j]) ** 2) * pmask[j]).sum() / pmask[j].sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(xe); out.append(o[0].cpu().numpy() + ym)
            if pump: pout.append(o[4].cpu().numpy() * psd + pmu)
    return np.mean(out, 0), (np.mean(pout, 0) if pout else None)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
KINDS = ("B", "BP", "T", "TP"); PRED = {k: {} for k in KINDS}; PP = {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind in KINDS:
            p, pp = fit_predict3(tr, te, kind)
            for j, r in enumerate(te):
                PRED[kind][(v, r)] = p[j]
                if pp is not None: PP[(kind, v, r)] = pp[j]
    print("done", v, flush=True)
res, blk = [], {}
PAIRS = {"BP": "B", "TP": "T"}
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    key = (G.rec.str[:3] + "_" + (G.day // 5).astype(str)).values; BL = {}
    for kind in KINDS:
        m = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m; BL[kind] = bl
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p2=rm((m - y)[p2]), alone_p2_normal=rm((m - y)[p2 & ~hi]), alone_p2_high=rm((m - y)[p2 & hi]),
                        blend_SSE=float(((bl - y) ** 2).sum()), blend_p2=rm((bl - y)[p2])))
    if v == "P2LOO":
        for x_, b_ in PAIRS.items(): blk.setdefault(x_, []).append(pd.Series((BL[x_] - y) ** 2 - (BL[b_] - y) ** 2).groupby(key).mean())
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
PB = .025 / 2; ok = {}
for x_, b_ in PAIRS.items():
    a = S[S.kind == x_].groupby("val").alone.mean() < S[S.kind == b_].groupby("val").alone.mean(); ca = bool(a.all())
    d = S[S.kind == x_].set_index(["val", "seed"]).blend_SSE - S[S.kind == b_].set_index(["val", "seed"]).blend_SSE; cb = bool((d < 0).all())
    xx = pd.concat(blk[x_], axis=1).mean(axis=1).values; mb = xx[np.random.default_rng(20261226).integers(0, len(xx), (20000, len(xx)))].mean(1); pw = float((mb >= 0).mean())
    ok[x_] = ca and cb and pw < PB
    print(f"{x_} vs {b_}: (a) alone lower all validators {ca} {a.to_dict()} | (b) blend dSSE<0 cells {int((d < 0).sum())}/9 | (c) P2LOO P(worse) {pw:.4f} (< {PB}) -> {'PASS' if ok[x_] else 'FAIL'}")
# 서술: 관수 머리가 F32 에서 실제로 배워지는지(F32 기록 5분할 OOF, 한 번), F13·F47 추정 펌프와 하루 EC 상관
rng = np.random.default_rng(0); fo = rng.permutation(M) % 5; cs = []
tr_all = LABREC
for k_ in range(5):
    _, pp = fit_predict3(tr_all, [], "BP", f32_idx=(I32[fo != k_], I32[fo == k_]))
    tv = P32[fo == k_]
    for c in range(2):
        mm = np.isfinite(tv[..., c]); cs.append(dict(ch=("pump", "valve")[c], r=np.corrcoef(pp[..., c][mm], tv[..., c][mm])[0, 1]))
print("irrigation head F32 OOF corr (all hours):", pd.DataFrame(cs).groupby("ch").r.mean().round(3).to_dict())
pr = pd.DataFrame([dict(rec=r, pump=float(np.mean(p[:, 0])), ec=float(DM[idx[r]])) for (k_, v, r), p in PP.items() if k_ == "BP" and v == "P2LOO"])
print("F13/F47 P2LOO: corr(est. daily pump, daily EC) =", round(float(pr[["pump", "ec"]].corr().iloc[0, 1]), 3), "n", len(pr))
print("MT3:", {k: ("PASS" if ok[k] else "FAIL") for k in ok})
pd.to_pickle(dict(PRED=PRED, PP=PP), os.path.join(env.LOCAL, "mt3_preds_v1.pkl"))
