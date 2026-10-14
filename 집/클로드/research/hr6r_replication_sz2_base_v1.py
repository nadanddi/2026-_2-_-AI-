# -*- coding: utf-8 -*-
"""HR6R (2026-10-15 집 클로드; 사용자 취침 중 '뭐라도 마구 해봐').  사전 고정(커밋 후 실행).  EC.
HR6(6.916 예정; 059f9df7): HR5 + 링커 어제 EC 특징 — 주 6/6칸 개선이나 P2LOO P .14 FAIL.  같은 HR6 를 바꾸지 않고 독립 기본 예측에서 재판정.
기본 예측 = SZ2 체크포인트(JOB001: 새 R3 시드 1357·2468·3579, 새 PFN 문맥, SZ2 폴드 배치 DIAG10(2차 포함 폴드)·EL1·P2LOO) 의 fin_SEA_s (= 지금 EC sg 단계와 같은 꼴).
HR5·HR6 보정기: HR6 스크립트와 같은 특징·학습 행(cp2 base_oof)·숨김(검증 + 같은 농장 ±1 + 다른 농장 ±3)·λ .5·TabPFN 설정; 링커 어제 EC 조건도 같음(검증 기록 정답 안 씀).
판정(EC 분야 배정 규칙, HR6 vs HR5): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
서술: 보통/고EC, 장부 제외, HR5·HR6 의 fin_SEA 대비 %.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hr6r_replication_sz2_base_v1.py
"""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr6_tabpfn_dayresid_anchor_feats_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr6_tabpfn_dayresid_anchor_feats_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "hr6r_replication_sz2_base_v1.py")
CK = os.path.join(env.LOCAL, "sz2_ckpt"); SEEDS = (1357, 2468, 3579); rid3 = lambda f, d: "%s_%03d" % (f, d)
def mk(): return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True)
rows = []
for fpath in sorted(glob.glob(os.path.join(CK, "*.csv"))):
    G = pd.read_csv(fpath); v, k = G.validator.iloc[0], int(G.validation_fold.iloc[0]); G["rec"] = [rid3(a, b) for a, b in zip(G.farm, G.day)]; val = set(G.rec)
    hide = np.zeros(len(TRK), bool)
    for r in val:
        f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTHER[f]) & (np.abs(tnum - n) <= 3))
    tr = ~hide; M5, M6 = {}, {}
    for h in HS:
        X6 = np.concatenate([TF[h], np.stack([extra(r, TR[r][0], h, val) for r in TRK])], 1)
        m = mk(); m.fit(TF[h][tr], TD[tr]); M5[h] = m; m = mk(); m.fit(X6[tr], TD[tr]); M6[h] = m
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour")
        if len(g) != 24 or r not in INP: continue
        for s in SEEDS:
            base = g[f"fin_SEA_{s}"].values
            Q = np.stack([feat(INP[r], base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS])); E = np.stack([extra(r, base, h, val) for h in HS])
            d5 = {h: float(M5[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}; d6 = {h: float(M6[h].predict(np.concatenate([Q[j], E[j]])[None])[0]) for j, h in enumerate(HS)}
            D5 = np.array([d5[max(x for x in HS if x <= hh)] for hh in g.hour.values]); D6 = np.array([d6[max(x for x in HS if x <= hh)] for hh in g.hour.values])
            rows.append(pd.DataFrame(dict(validator=v, seed=s, rec=r, farm=g.farm.values, day=g.day.values, hour=g.hour.values, y=g.sub_ec.values, base=base, hr5=base + .5 * D5, hr6=base + .5 * D6, used=bool(E[0, 2]))))
    print(v, k, "done", flush=True)
B = pd.concat(rows, ignore_index=True); B["d"] = (B.hr6 - B.y) ** 2 - (B.hr5 - B.y) ** 2; B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); print("records with yesterday EC used:", B.groupby("validator").used.mean().round(3).to_dict())
res = []
for (v, sd), g in B.groupby(["validator", "seed"]):
    p2 = g.p2.values; f = lambda m: 100 * (rm((g.hr6 - g.y)[m]) / rm((g.hr5 - g.y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE_p2=float(g.d[p2].sum()), p2=f(p2), all=f(np.ones(len(g), bool)), p2_normal=f(p2 & ~g.hi.values), p2_high=f(p2 & g.hi.values),
                    hr5_vs_base=100 * (rm(g.hr5 - g.y) / rm(g.base - g.y) - 1), hr6_vs_base=100 * (rm(g.hr6 - g.y) / rm(g.base - g.y) - 1), X0_p2=float(g.d[p2 & ~g.rec.isin(LEDGER).values].sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(v, p2only):
    g = B[(B.validator == v) & (B.p2 if p2only else True)]; s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20270101).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot("P2LOO", True)
RD = B[B.validator.isin(["P2LOO", "EL1"]) & B.p2].groupby(["validator", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum())
_, pb = boot("DIAG10", False); c3 = not (B[B.validator == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
print(f"\n(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("HR6R:", "ADOPT" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
B.to_csv(os.path.join(HERE, "hr6r_rows_v1.csv.gz"), index=False)
