# -*- coding: utf-8 -*-
"""CR1 (2026-10-13 집 클로드; 사용자: '온도·EC 는 이어지는 기록이 중요 → RNN 으로, 온도부터').  사전 고정(커밋 후 실행).  온도.
'같은 동 사슬 GRU' — 기록 번호 순서(대부분 다른 동)가 아니라 짝찾기로 찾은 같은 동 전날을 앞에 이어 48단계 시퀀스.
  전날(1~24단계): 전날 기록의 실내온도·실내습도·외기온·일사·난방·보온커튼·천창 + 전날 배지온도 정답(학습 기록이고 그 폴드에서 숨김이 아니면; 아니면 0 + 표시)
                 + '전날 있음'·'정답 보임' 표시.  전날 = v15R(거울 G0) 1순위가 정답 기록이고 (p0 ≥ .8 또는 HP2 같은 짝) → 아니면 v7 확신 앵커 짝 → 아니면 없음.
  오늘(25~48단계): 같은 입력 + 기준 예측 q_h + 시각.  한 방향 GRU 이므로 h 시 출력은 오늘 0..h 와 전날만 봄(규정: 같은 온실 해당·이전 + 학습 정답).
  출력 = 오늘 각 시각 보정량(목표 = y − 기준).
모델: GRU(입력 →32, 1층) → Linear(1), Adam lr 2e-3·weight decay 1e-4·80 에폭·배치 64, 시드 1/2/3 평균, GPU.  특징 표준화·NaN→0 은 학습 폴드 기록으로.
학습: 배치(DIAG10, EL1, EXT10, EXT12, F~L)마다 폴드별 교차 적합 — 폴드 k 기록을 맞힐 때 폴드 k 밖 기록으로 학습(정답 보임 판정도 폴드 k 기준), 셀 평균 기준으로 학습해 각 셀에 같은 보정 적용.
안(2개, α .0125):  G2 = 앵커 전 셀 q + GRU(목표 y − q)   (앵커 공식을 GRU 로 대체);   G1 = 14회차형(q + v7 앵커 + TNA8 앵커) + GRU(목표 y − 앵커판).
기준 = 14회차형(셀 + v7 앵커 + TNA8 앵커, 측정 감쇠).  판정(온도 범위 규칙 ALL): 주 DIAG10·EXT10 모두 4칸 개선 & 농장×5일 블록 부트스트랩 P(악화) < .0125;
  나머지 배치는 칸 평균 악화이면서 블록 '후보가 나음' 비율 < .0125 이면 FAIL.
"""
import os, sys, importlib.util
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s_ = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s_); s_.loader.exec_module(A7)
A7.BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read()
_ns = {"__file__": os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), "__name__": "tb1src"}; exec(src[:src.index("PE15 = ")], _ns)
hp2, mirror_lab, ST, idx, TRAIN, FARM, NUM, N = (_ns[k] for k in ("hp2", "mirror_lab", "ST", "idx", "TRAIN", "FARM", "NUM", "N"))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261122); DEV = "cuda" if torch.cuda.is_available() else "cpu"
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
L7f = L7[L7.prev_is_labelled.astype(bool) & L7.anchor_st23.notna() & (L7.steps_to_anchor == 1) & L7.rec.isin(v7set)]
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
PREV = {}; TNA8 = {}
for r in PM.index:
    pv, p0, lb = PM.at[r, "prev0"], PM.at[r, "p0"], PM.at[r, "lab0"]
    if isinstance(pv, str) and bool(lb) and (p0 >= .8 or hp2(idx[r], mirror_lab(r)) == pv):
        PREV[r] = pv
        if r not in v7set: TNA8[r] = pv
for r, pv in zip(L7f.rec, L7f.prev): PREV.setdefault(r, pv)
print(f"predecessors: {len(PREV)} records | TNA8 extra anchors (mirror): {len(TNA8)}", flush=True)
add = pd.DataFrame(dict(rec=list(TNA8), prev=list(TNA8.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8.values()], dT_air=0.0, steps_to_anchor=1))
LC = pd.concat([L7f, add], ignore_index=True)
V = ["in_temp", "in_hum", "out_temp", "out_rad", "act_heating", "act_thermal", "act_vent"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CUBE = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in V], 2)   # [N,24,7]
class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.g = nn.GRU(d, 32, batch_first=True); s.o = nn.Linear(32, 1)
    def forward(s, x): return s.o(s.g(x)[0]).squeeze(-1)
def build(recs, qmap, hidden):
    """recs: list of record ids; qmap: rec -> base array[24]; hidden: set of hidden recs (labels not visible)"""
    Xs = np.zeros((len(recs), 48, len(V) + 1 + 3 + 2), np.float32)
    for k, r in enumerate(recs):
        b = idx[r]; pv = PREV.get(r)
        if pv is not None:
            a = idx[pv]; Xs[k, :24, :len(V)] = CUBE[a]
            vis = TRAIN[a] and pv not in hidden
            if vis: Xs[k, :24, len(V)] = ST[a]
            Xs[k, :24, len(V) + 1] = 1.0; Xs[k, :24, len(V) + 2] = float(vis)
        Xs[k, 24:, :len(V)] = CUBE[b]; Xs[k, 24:, len(V) + 3] = qmap[r]; Xs[k, 24:, len(V) + 4] = np.arange(24) / 23
    return Xs
def fit_predict(Xtr, Ytr, Xte):
    mu = np.nanmean(Xtr.reshape(-1, Xtr.shape[2]), 0); sd = np.nanstd(Xtr.reshape(-1, Xtr.shape[2]), 0) + 1e-6
    for flag in (len(V) + 1, len(V) + 2): mu[flag] = 0; sd[flag] = 1
    z = lambda A: np.nan_to_num((A - mu) / sd).astype(np.float32)
    xt, yt, xe = torch.tensor(z(Xtr)).to(DEV), torch.tensor(np.nan_to_num(Ytr).astype(np.float32)).to(DEV), torch.tensor(z(Xte)).to(DEV)
    mt = torch.tensor(np.isfinite(Ytr).astype(np.float32)).to(DEV); preds = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed); net = Net(xt.shape[2]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        g = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(80):
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), 64):
                j = perm[i:i + 64]; out = net(xt[j])[:, 24:]; loss = ((out - yt[j]) ** 2 * mt[j]).sum() / mt[j].sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): preds.append(net(xe)[:, 24:].cpu().numpy())
    return np.mean(preds, 0)
res = []; blocks = {}
for L, (M, cells) in A7.A2.A1.frames().items():
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    qbar = np.mean(cells, 0); abar = np.mean(anc, 0)
    recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    def per_rec(arr): return {r: np.full(24, np.nan) if len(pos[r]) == 0 else np.array([arr[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    Q, A, Y = per_rec(qbar), per_rec(abar), per_rec(y)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; corr2 = {}; corr1 = {}
    for k in sorted(set(fold_of.values())):
        te = [r for r in recs if fold_of[r] == k]; tr = [r for r in recs if fold_of[r] != k]; hid = set(te)
        for base, store in ((Q, corr2), (A, corr1)):
            Xtr = build(tr, base, hid); Ytr = np.stack([Y[r] - base[r] for r in tr]); Xte = build(te, base, hid)
            P = fit_predict(Xtr, Ytr, Xte)
            for r, p in zip(te, P): store[r] = p
    def apply(arrs, corr):
        outs = []
        for c in arrs:
            n = c.copy()
            for r in recs:
                ix = pos[r]; n[ix] = c[ix] + corr[r][h[ix]]
            outs.append(n)
        return outs
    g2 = apply(cells, corr2); g1 = apply(anc, corr1)
    for ci in range(len(cells)):
        for a, new in (("G2", g2[ci]), ("G1", g1[ci])):
            base = anc[ci]; p2 = M.day.values >= 179
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
S = pd.DataFrame(res); pd.set_option("display.width", 220)
print(S.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
print(S.groupby(["arm", "layout"]).p2.mean().round(3).unstack(0).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a in ("G2", "G1"):
    ok = True; msg = []
    for L in sorted(set(S.layout)):
        cells_ = S[(S.arm == a) & (S.layout == L)].pct; pw, pb = bootP(L, a)
        if L in ("DIAG10", "EXT10"):
            c = bool((cells_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((cells_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = cells_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {cells_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== CR1 {a}:"); print("\n".join("   " + m for m in msg)); print(f"CR1 {a}:", "ADOPT" if ok else "FAIL")
