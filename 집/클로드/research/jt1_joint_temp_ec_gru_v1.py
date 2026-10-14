# -*- coding: utf-8 -*-
"""JT1 (2026-10-14 집 클로드; 사용자: 'MTL 과 RNN 결합 — 1번(온도 GRU 와 EC MTL 을 한 신경망으로) 진행해봐').  사전 고정(커밋 후 실행).  EC + 온도.
박사 형 구조: input → (EC, 배지온도, …) → 상호작용 층 → (EC, 배지온도, …).  여기선 배지온도를 보조가 아니라 '주 출력'(손실 가중 1)으로.
모델(몸통 = MT1·MT2 와 같은 하루 GRU 2층 128, 자기 기록 0..h 입력만 → 5점 유형; 입력·표준화·25 에폭·배치 16·내부 시드 1·2 동일):
  JT = GRU → 1차 머리 [EC1, TEMP1] → mix MLP([z, 1차]) → [EC2, TEMP2];  손실 = EC1 + EC2 + TEMP1 + TEMP2 (배지온도는 폴드 학습 기록으로 표준화)
  B  = MT1(EC 주 + 남은 하루 입력 보조)            — EC 쪽 기준
  TG = JT 와 같은 망, EC 손실 끔(배지온도만)        — 온도 쪽 기준(같은 구조의 온도 단독 GRU)
검증: ec_rv1 CUR 배치(DIAG10·P2LOO·EL1, MT1 숨김 규칙). 온도 정답은 같은 검증 기록의 sub_temp.
판정:
 [EC] JT vs B (MT2 와 같은 규칙, 단일 비교 α .025): (a) 단독 EC RMSE 세 검증기 모두 낮음 AND (b) 섞기(0.9·sg + 0.1·X) dSSE < 0 9칸 모두
      AND (c) P2LOO 섞기 행 블록 부트스트랩 P(악화) < .025.
 [온도 관문] JT 온도 머리 vs TG: 단독 온도 RMSE 세 검증기 모두 낮음 AND P2LOO 행 dSSE 블록 부트스트랩 P(악화) < .025
      → 통과하면 'JT2: v23 온도 파이프라인에서 V3(번호 순서 48시간 창, 5점 아님) 자리를 JT 온도 머리로' 를 별도 사전 고정(채택 판정 아님).
서술: 2차 보통/고EC, 추운 행(in_temp < 8) 온도 RMSE, 참고로 v23 온도 사슬과 비교는 안 함(배치 다름).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u jt1_joint_temp_ec_gru_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt2_mtl_expert_informed_aux_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "jt1_joint_temp_ec_gru_v1.py")
def fit_joint(tr, te, kind):
    it, ie = np.array([idx[r] for r in tr]), np.array([idx[r] for r in te])
    A_ = CUBE[it]; mu = np.nanmean(A_.reshape(-1, A_.shape[2]), 0); sd = np.nanstd(A_.reshape(-1, A_.shape[2]), 0) + 1e-6
    def prep(I):
        Z = (CUBE[I] - mu) / sd; m = np.isnan(CUBE[I][..., 4:5]).astype(np.float32); return torch.tensor(np.concatenate([np.nan_to_num(Z), m], 2), device=DEV)
    xt, xe = prep(it), prep(ie); Y = EC[it].astype(np.float32); ym = float(np.nanmean(Y))
    yt = torch.tensor(np.nan_to_num(Y - ym), device=DEV); mt = torch.tensor(np.isfinite(Y).astype(np.float32), device=DEV)
    T_ = TMP[it]; tmu, tsd = float(np.nanmean(T_)), float(np.nanstd(T_)) + 1e-6
    tt = torch.tensor(np.nan_to_num((T_ - tmu) / tsd)[..., None], device=DEV); tm = torch.tensor(np.isfinite(T_).astype(np.float32)[..., None], device=DEV)
    we = 0.0 if kind == "TG" else 1.0; oe, ot = [], []
    for seed in (1, 2):
        torch.manual_seed(seed); net = NetB(xt.shape[2], 1).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); g = torch.Generator().manual_seed(seed)
        for ep in range(25):
            perm = torch.randperm(len(xt), generator=g).to(DEV)
            for b in range(0, len(xt), 16):
                j = perm[b:b + 16]; mj = mt[j]; e2, e1, a1, a2 = net(xt[j])
                ec_l = lambda e: (((e - yt[j]) ** 2) * mj).sum() / mj.sum().clamp(min=1); t_l = lambda a: (((a - tt[j]) ** 2) * tm[j]).sum() / tm[j].sum().clamp(min=1)
                loss = we * (ec_l(e2) + ec_l(e1)) + t_l(a1) + t_l(a2)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            e2, _, _, a2 = net(xe); oe.append(e2.cpu().numpy() + ym); ot.append(a2[..., 0].cpu().numpy() * tsd + tmu)
    return np.mean(oe, 0), np.mean(ot, 0)
def fit_B(tr, te):
    p, _, _ = fit_predict2(tr, te, "B"); return p
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
def nbr(te): return {r for r in LABREC for t in te if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
PE = {"B": {}, "JT": {}}; PT = {"JT": {}, "TG": {}}
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())]
    for kf, Gk in G.groupby("fold"):
        te = sorted(set(Gk.rec)); hide = set(te) | (nbr(te) if v == "EL1" else set()); tr = [r for r in LABREC if r not in hide]
        pb = fit_B(tr, te); pe, pt = fit_joint(tr, te, "JT"); _, pg = fit_joint(tr, te, "TG")
        for j, r in enumerate(te): PE["B"][(v, r)] = pb[j]; PE["JT"][(v, r)] = pe[j]; PT["JT"][(v, r)] = pt[j]; PT["TG"][(v, r)] = pg[j]
    print("done", v, flush=True)
pd.set_option("display.width", 220); res, blk = [], []
for (v, sd_), G in O[O.validator.isin(["DIAG10", "P2LOO", "EL1"])].groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    key = (G.rec.str[:3] + "_" + (G.day // 5).astype(str)).values; BL = {}
    for kind in ("B", "JT"):
        m = np.array([PE[kind][(v, r)][h] for r, h in zip(G.rec, G.hour)]); bl = .9 * sg + .1 * m; BL[kind] = bl
        res.append(dict(val=v, seed=sd_, kind=kind, alone=rm(m - y), alone_p2=rm((m - y)[p2]), alone_p2_normal=rm((m - y)[p2 & ~hi]), alone_p2_high=rm((m - y)[p2 & hi]), blend_SSE=float(((bl - y) ** 2).sum())))
    if v == "P2LOO": blk.append(pd.Series((BL["JT"] - y) ** 2 - (BL["B"] - y) ** 2).groupby(key).mean())
S = pd.DataFrame(res); print("[EC]"); print(S.round(4).to_string(index=False))
a = S[S.kind == "JT"].groupby("val").alone.mean() < S[S.kind == "B"].groupby("val").alone.mean(); ca = bool(a.all())
d = S[S.kind == "JT"].set_index(["val", "seed"]).blend_SSE - S[S.kind == "B"].set_index(["val", "seed"]).blend_SSE; cb = bool((d < 0).all())
xx = pd.concat(blk, axis=1).mean(axis=1).values; mb = xx[np.random.default_rng(20261226).integers(0, len(xx), (20000, len(xx)))].mean(1); pw = float((mb >= 0).mean())
ecok = ca and cb and pw < .025
print(f"[EC] JT vs B: (a) {ca} {a.to_dict()} | (b) cells {int((d < 0).sum())}/9 | (c) P2LOO P(worse) {pw:.4f} -> {'PASS' if ecok else 'FAIL'}")
# 온도 (검증기별 기록 1벌 — 시드는 EC 기본 예측에만 있으므로 최소 시드 행만)
tr_, tb = [], []
for v in ("DIAG10", "P2LOO", "EL1"):
    G = O[(O.validator == v) & (O.seed == O.seed.min())].sort_values(["rec", "hour"]); recs = G.rec.values; hrs_ = G.hour.values
    y = np.array([TMP[idx[r]][h] for r, h in zip(recs, hrs_)]); ok_ = np.isfinite(y); cold = np.array([RAW[idx[r]][h, USE.index("in_temp")] < 8 for r, h in zip(recs, hrs_)])
    pj = np.array([PT["JT"][(v, r)][h] for r, h in zip(recs, hrs_)]); pg = np.array([PT["TG"][(v, r)][h] for r, h in zip(recs, hrs_)])
    p2 = G.day.values >= 179
    tr_.append(dict(val=v, JT=rm((pj - y)[ok_]), TG=rm((pg - y)[ok_]), JT_p2=rm((pj - y)[ok_ & p2]) if (ok_ & p2).any() else np.nan, TG_p2=rm((pg - y)[ok_ & p2]) if (ok_ & p2).any() else np.nan,
                    JT_cold=rm((pj - y)[ok_ & cold]) if (ok_ & cold).any() else np.nan, TG_cold=rm((pg - y)[ok_ & cold]) if (ok_ & cold).any() else np.nan))
    if v == "P2LOO":
        key = (pd.Series(recs).str[:3] + "_" + (G.day.values // 5).astype(str)).values
        tb = pd.Series(((pj - y) ** 2 - (pg - y) ** 2)[ok_]).groupby(key[ok_]).mean().values
T = pd.DataFrame(tr_); print("[TEMP]"); print(T.round(4).to_string(index=False))
mb = tb[np.random.default_rng(20261226).integers(0, len(tb), (20000, len(tb)))].mean(1); pwt = float((mb >= 0).mean()); ta = bool((T.JT < T.TG).all())
print(f"[TEMP gate] JT temp head vs TG: lower all validators {ta} | P2LOO P(worse) {pwt:.4f} -> {'PASS (pre-register JT2)' if ta and pwt < .025 else 'FAIL'}")
print("JT1:", dict(EC="PASS" if ecok else "FAIL", TEMP_gate="PASS" if (ta and pwt < .025) else "FAIL"))
pd.to_pickle(dict(PE=PE, PT=PT), os.path.join(env.LOCAL, "jt1_preds_v1.pkl"))
