# -*- coding: utf-8 -*-
"""MT2 (2026-10-14 집 클로드; 사용자: '박사 형이 MTL 에 붙일 때 연구사 코멘트 참고 — 우리도 txt 로 가진 것').  사전 고정(커밋 후 실행).  EC.
근거: 연구사 답변(공용/전문가_질의응답/2026-10-09, 6.467) — 'EC 는 외부 환경보다 식물체 상태·관수 조건에 좌우',
  '2.0 이상은 생식생장, 0.5 는 영양생장(동마다 생육 단계 다름)', '관수 물 온도에 따라 배지온도가 떨어짐'.
  MT1(6.894) 은 보조 과제가 환경 입력(실내 온도·CO₂·구동기)의 남은 하루 평균 → 코멘트와 방향이 다름.
안 (몸통 = MT1 과 같은 하루 GRU 2층 128, 입력·표준화·학습 설정·내부 시드 1·2 동일):
  B  = MT1 그대로(보조 = 남은 하루 입력 7종 평균; 1차 머리 → mix → 2차 머리)                       — 기준
  T  = B 구조, 보조만 '그 시각 배지온도(sub_temp 정답), 배지온도 − 실내온도' 2개로 바꿈(관수 흔적 학습용)
  R  = 상태 관문: 몸통 → 상태 logit s(보조 목표 = 그 기록 하루 평균 EC ≥ 1.0, 학습 정답), 전문가 E_hi·E_lo(각 128→64→1),
       EC = σ(s)·E_hi + (1 − σ(s))·E_lo;  손실 = EC + .5·BCE(s)
  TR = R + 몸통에서 배지온도 보조 머리 2개(T 와 같은 목표, 가중 .5)
  보조 목표는 학습 기록 정답만(평가 행 입력·정답 안 씀); 평가 시 입력은 그 기록 자신의 0..h 입력만(단방향 GRU) → 5점 유형.
  배지온도 보조 목표 표준화는 폴드 학습 기록으로.  온실 ID 안 씀.
검증: MT1 과 같은 ec_rv1 CUR 배치(DIAG10·P2LOO·EL1, 숨김 규칙 동일), sg 와 0.9·sg + 0.1·X 섞기(비율 고정).
판정(안마다, 본페로니 k = 3 → .025/3 = .0083):  X 가 B 보다 낫다 =
  (a) 단독 RMSE 가 세 검증기 모두 B 보다 낮음  AND  (b) 섞기 dSSE(X − B) < 0 이 9칸(3 검증기 x 3 시드) 모두
  AND (c) P2LOO 섞기 행 dSSE(X − B) 의 농장 x 5일 블록 부트스트랩 P(악화) < .0083.
  통과한 안이 여럿이면 P2LOO 2차 섞기 RMSE 가 가장 낮은 안을 NHM 의 MT1 대체 후보로 보고(그 대체는 별도 정식 시험 필요).
서술: 단독 2차 보통/고EC, 상태 머리 AUC(고EC 날 판별, 시각 23 출력), 배지온도 보조 상관.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u mt2_mtl_expert_informed_aux_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("def fit_predict")])
__file__ = os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py")
from sklearn.metrics import roc_auc_score
_Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); _Y = _Y[_Y.row_id.str[:3].isin(["F13", "F47"])]; _Y["rec"] = _Y.row_id.str[:7]; _Y["h"] = _Y.row_id.str[8:].astype(int)
TMP = _Y.pivot(index="rec", columns="h", values="sub_temp").reindex(index=[rid(i) for i in range(N)], columns=range(24)).values.astype(np.float32)
TAUX = np.stack([TMP, TMP - RAW[:, :, USE.index("in_temp")]], 2)   # [N, 24, 2]
STATE = (DM >= 1.0).astype(np.float32)
class NetR(torch.nn.Module):
    def __init__(s, d, kt):
        super().__init__(); s.g = torch.nn.GRU(d, 128, num_layers=2, batch_first=True); s.s = torch.nn.Linear(128, 1)
        s.eh = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)); s.el = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        s.t = torch.nn.Linear(128, kt) if kt else None
    def forward(s, x):
        z = s.g(x)[0]; sl = s.s(z).squeeze(-1); p = torch.sigmoid(sl)
        e = p * s.eh(z).squeeze(-1) + (1 - p) * s.el(z).squeeze(-1)
        return e, sl, (s.t(z) if s.t is not None else None)
def fit_predict2(tr, te, kind):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A_ = CUBE[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    F = FUT[it] if kind == "B" else TAUX[it]; k = F.shape[2]
    fm, fs = np.nanmean(F.reshape(-1, k), 0), np.nanstd(F.reshape(-1, k), 0) + 1e-6; Fz = (F - fm) / fs
    ft = torch.tensor(np.nan_to_num(Fz).astype(np.float32), device=DEV); fmask = torch.tensor(np.isfinite(Fz).astype(np.float32), device=DEV)
    st = torch.tensor(STATE[it], device=DEV)[:, None].expand(-1, 24)
    out, sout, tout = [], [], []
    for seed in (1, 2):
        torch.manual_seed(seed)
        net = (NetB(xt.shape[2], k) if kind in ("B", "T") else NetR(xt.shape[2], 2 if kind == "TR" else 0)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; mj = mt[j]; fj, fmj = ft[j], fmask[j]; ec_l = lambda e: (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1)
                aux_l = lambda a: (((a - fj) ** 2) * fmj).sum() / fmj.sum().clamp(min=1)
                if kind in ("B", "T"):
                    e2, e1, a1, a2 = net(xt[j]); loss = ec_l(e2) + ec_l(e1) + .5 * (aux_l(a1) + aux_l(a2))
                else:
                    e, sl, ta = net(xt[j]); loss = ec_l(e) + .5 * torch.nn.functional.binary_cross_entropy_with_logits(sl, st[j])
                    if ta is not None: loss = loss + .5 * aux_l(ta)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(xe); out.append(o[0].cpu().numpy() + ym)
            if kind in ("R", "TR"): sout.append(torch.sigmoid(o[1]).cpu().numpy())
            if kind == "T": tout.append(o[3].cpu().numpy() * fs + fm)
            if kind == "TR": tout.append(o[2].cpu().numpy() * fs + fm)
    return np.mean(out, 0), (np.mean(sout, 0) if sout else None), (np.mean(tout, 0) if tout else None)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
KINDS = ("B", "T", "R", "TR"); PRED = {k: {} for k in KINDS}; SP, TP = {}, {}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        for kind in KINDS:
            p, s_, t_ = fit_predict2(tr, te, kind)
            for j, r in enumerate(te):
                PRED[kind][(v, r)] = p[j]
                if s_ is not None: SP[(kind, v, r)] = s_[j]
                if t_ is not None: TP[(kind, v, r)] = t_[j]
    print("done", v, flush=True)
res, blk = [], {}
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    key = (G.rec.str[:3] + "_" + (G.day // 5).astype(str)).values; BL = {}
    for kind in KINDS:
        m = np.array([PRED[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m; BL[kind] = bl
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p2=rm((m - y)[p2]), alone_p2_normal=rm((m - y)[p2 & ~hi]), alone_p2_high=rm((m - y)[p2 & hi]),
                        blend_SSE=float(((bl - y) ** 2).sum()), blend_p2=rm((bl - y)[p2])))
    if v == "P2LOO":
        for kind in KINDS[1:]: blk.setdefault(kind, []).append(pd.Series((BL[kind] - y) ** 2 - (BL["B"] - y) ** 2).groupby(key).mean())
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
PB = .025 / 3; best, ok = None, {}
for kind in KINDS[1:]:
    a = S[S.kind == kind].groupby("val").alone.mean() < S[S.kind == "B"].groupby("val").alone.mean(); ca = bool(a.all())
    d = S[S.kind == kind].set_index(["val", "seed"]).blend_SSE - S[S.kind == "B"].set_index(["val", "seed"]).blend_SSE; cb = bool((d < 0).all())
    x = pd.concat(blk[kind], axis=1).mean(axis=1).values; mb = x[np.random.default_rng(20261226).integers(0, len(x), (20000, len(x)))].mean(1); pw = float((mb >= 0).mean())
    ok[kind] = ca and cb and pw < PB
    print(f"{kind}: (a) alone lower all validators {ca} {a.to_dict()} | (b) blend dSSE<0 cells {int((d < 0).sum())}/9 | (c) P2LOO P(worse) {pw:.4f} (< {PB:.4f}) -> {'PASS' if ok[kind] else 'FAIL'}")
P2 = S[S.val == "P2LOO"].groupby("kind").blend_p2.mean(); cand = [k for k in ok if ok[k]]
if cand: best = min(cand, key=lambda k: P2[k])
for kind in ("R", "TR"):
    yy, ss = [], []
    for (k_, v, r), s_ in SP.items():
        if k_ == kind and v == "P2LOO": yy.append(STATE[idx[r]]); ss.append(s_[23])
    if len(set(yy)) > 1: print(f"{kind} state head AUC (P2LOO, hour 23): {roc_auc_score(yy, ss):.3f}  (n={len(yy)}, high={int(sum(yy))})")
for kind in ("T", "TR"):
    cs = [np.corrcoef(t[:, 0][np.isfinite(TMP[idx[r]])], TMP[idx[r]][np.isfinite(TMP[idx[r]])])[0, 1] for (k_, v, r), t in TP.items() if k_ == kind and v == "P2LOO" and np.isfinite(TMP[idx[r]]).sum() > 5]
    print(f"{kind} sub_temp aux corr within record (P2LOO median): {np.nanmedian(cs):.3f}")
print("MT2:", {k: ("PASS" if ok[k] else "FAIL") for k in ok}, "-> best:", best)
pd.to_pickle(dict(PRED=PRED, SP=SP), os.path.join(env.LOCAL, "mt2_preds_v1.pkl"))
