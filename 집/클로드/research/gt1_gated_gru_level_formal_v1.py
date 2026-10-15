# -*- coding: utf-8 -*-
"""GT1 (2026-10-15 집 클로드; 사용자: EVX2 분석 후 'a, b 진행해봐').  사전 고정(커밋 후 실행).  EC.
EVX2(사후 서술): GRU(EVY) 는 '어제 링크가 있는 날' 단독 RMSE .159 vs 기본 .227(하루 수준에서 강함), 링크 없는 날 .38 vs .26(약함);
  하루 안 모양은 나무 기본이 강함(.072 vs .12) → 고정 10% 섞기는 강한 곳 이득을 버리고 약한 곳에서 손해.
EVY(evx 스크립트와 같은 코드·학습) 를 SZ2 폴드(62)에 다시 학습; 링크 = 패키지 25 MASK 세계 거울 링커(v15r_mirror_masked_pkg25.csv) p0 ≥ .5 (26번 규칙),
  어제 EC 채널 = 짝이 정답 있는 학습 기록이고 그 폴드 숨김(검증 + 같은 농장 ±1 + 다른 농장 ±3)이 아닐 때만.
NH_s = pre_SEA_s + .5·HR5 (nh1 저장 행에서 되살림), F5E_s = .9·NH_s + .1·EV3L (기준, NH2 와 같음).
안(링크 = 그 기록에 p0 ≥ .5 짝이 있음, 0시에 정해짐 → 인과; 비율 .5 는 사후 최적 .7~.9 보다 보수적으로 미리 고정):
  A = 링크 있으면 .5·NH + .5·EVY, 없으면 NH
  B = 링크 있으면 NH + .5·(cm_h(EVY) − cm_h(NH))  (cm_h = 그날 0..h 누적 평균: 하루 수준만 GRU 쪽으로, 하루 안 모양은 NH), 없으면 NH
판정(EC 분야 배정 규칙, 각 안 vs F5E, 본페로니 k 2 → .0125): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .0125 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
서술: 링크 있는/없는 기록 분해, 15회차형(SH)·구조판(S) 대비, 장부 제외.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u gt1_gated_gru_level_formal_v1.py
"""
import os, sys, pickle, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "evx_ev3l_mtl_and_yesterday_ec_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "evx_ev3l_mtl_and_yesterday_ec_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "gt1_gated_gru_level_formal_v1.py")
Lm = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); LINK = {r: p for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .5 and p != r and p in idx}
C48 = np.full((N, 48, CUBE.shape[2]), np.nan, np.float32); C48[:, 24:] = CUBE
for r, p in LINK.items():
    if r in idx: C48[idx[r], :24] = CUBE[idx[p]]
SEEDS = (1357, 2468, 3579); OTH = {"F13": "F47", "F47": "F13"}; PB = .025 / 2
CKS = os.path.join(env.LOCAL, "sz2_ckpt"); EF = os.path.join(env.LOCAL, "gt1_evy_v1.pkl"); EY = pickle.load(open(EF, "rb")) if os.path.exists(EF) else {}
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"))
for (v, k), G in R.groupby(["validator", "fold"]):
    if (v, k) in EY: continue
    te = sorted(set(G.rec)); vd = {(r[:3], int(r[4:])) for r in te}
    hide = {r for r in LABREC if any((r[:3] == f and abs(int(r[4:]) - d) <= 1) or (r[:3] == OTH[f] and abs(int(r[4:]) - d) <= 3) for f, d in vd)} | set(te)
    tr = [r for r in LABREC if r not in hide]; p = fit_x(tr, te, "EVY", hide); EY[(v, k)] = {r: p[j] for j, r in enumerate(te)}
    pickle.dump(EY, open(EF, "wb")); print("EVY", v, k, "done", flush=True)
MTp = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
mt = np.array([MTp[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
ey = np.array([EY[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); R["linked"] = R.rec.isin(LINK.keys()); lk = R.linked.values
R = R.sort_values(["validator", "fold", "rec", "hour"]); order = R.index.values
for s in SEEDS:
    nh = ((R[f"A1_{s}"] - .1 * mt[order]) / .9).values; R[f"F5E_{s}"] = .9 * nh + .1 * ev[order]; e_ = ey[order]; l_ = lk[order]
    R[f"A_{s}"] = np.where(l_, .5 * nh + .5 * e_, nh)
    gkey = (R.validator + "|" + R.fold.astype(str) + "|" + R.rec).values
    cm = lambda x: pd.Series(x).groupby(gkey).cumsum().values / (pd.Series(np.ones(len(x))).groupby(gkey).cumsum().values)
    R[f"B_{s}"] = np.where(l_, nh + .5 * (cm(e_) - cm(nh)), nh)
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}; pd.set_option("display.width", 240); VER = {}
print("records linked (share):", R.groupby("validator").apply(lambda g: g.groupby("rec").linked.first().mean()).round(3).to_dict())
for arm in ("A", "B"):
    rows, blocks, recd = [], {}, []
    for v in ("DIAG10", "EL1", "P2LOO"):
        g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; hi = g.hi.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; led = ~g.rec.isin(LEDGER).values; li = g.linked.values
        for s in SEEDS:
            a, n = g[f"F5E_{s}"].values, g[f"{arm}_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
            fx = lambda ref, m: 100 * (rm((n - y)[m]) / rm((g[ref].values - y)[m]) - 1) if m.any() else np.nan
            rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), vsF5E_all=fx(f"F5E_{s}", np.ones(len(y), bool)), vsF5E_p2=fx(f"F5E_{s}", p2),
                             p2_linked=fx(f"F5E_{s}", p2 & li), p2_unlinked=fx(f"F5E_{s}", p2 & ~li), p2_high=fx(f"F5E_{s}", p2 & hi), vsS_p2=fx(f"S_{s}", p2), vsSH_p2=fx(f"SH_{s}", p2),
                             vsSH_all=fx(f"SH_{s}", np.ones(len(y), bool)), X0_p2=float(d[p2 & led].sum())))
            blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
            for r_, dd in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd))
    S = pd.DataFrame(rows); print(f"\n=== {arm} vs F5E ==="); print(S.round(3).to_string(index=False))
    def boot(key_):
        x = pd.concat(blocks[key_], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20270106).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot(("P2LOO", "p2"))
    RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh_ = int((harmed > 1e-6).sum())
    dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
    VER[arm] = c1 and (pP < PB or nh_ == 0) and c3
    print(f"(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} (< {PB}) | harmed {nh_} | (3) DIAG10 mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'} => {arm}: {'ADOPT' if VER[arm] else 'FAIL'}")
    t = RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values(); print("P2LOO best/worst:", t.head(4).round(3).to_dict(), t.tail(4).round(3).to_dict())
print("\nGT1:", VER)
