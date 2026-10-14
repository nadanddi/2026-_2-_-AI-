# -*- coding: utf-8 -*-
"""HR6 (2026-10-15 집 클로드; 사용자 취침 중 '뭐라도 마구 해봐').  사전 고정(커밋 후 실행).  EC.
LA1(6.915): 링커 어제 EC 를 규칙(앵커)으로 덮으면 틀린 짝 하나가 하루를 망침.  C1(6.911): 진짜 어제 EC 를 알면 −40%.
→ 어제 정보를 규칙이 아니라 HR5 하루 수준 TabPFN 의 '특징'으로 넘겨, 얼마나 믿을지 TabPFN 이 학습 자료에서 배우게 한다.
HR6 = HR5(6.890, λ .5 고정, 기준 시각 h' 8개, 특징 45) + 특징 3개:
  AF_h' = A − (기본 0..h' 누적 평균),  A = 링커 짝(v15R 거울 prev0, p0 ≥ .3)의 23시 EC 정답;  AP = p0;  AH = A 를 썼으면 1.
  A 를 쓰는 조건: 짝이 정답 있는 학습 기록 & 잠금 아님 & 그 폴드의 검증 기록이 아님(학습 행·검증 행 똑같이).  못 쓰면 AF = 0, AH = 0 (AP 는 그대로).
비교 = 같은 폴드·같은 코드의 HR5 (λ .5).  기본 = ec_rv1 CUR sg(시드 47·1414·6464).
판정(EC 분야 배정 규칙, HR6 vs HR5): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 AND (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0);  (3) DIAG10 손해 감시.
  [링커를 쓰므로 EV 견고성이 원래 필요하나 ec_rv1 에 EV 행 없음 → 한계로 보고]
서술: 장부 제외 dSSE, 고EC/보통, A 를 쓴 기록 비율, 바뀐 기록 상위.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u hr6_tabpfn_dayresid_anchor_feats_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "hr6_tabpfn_dayresid_anchor_feats_v1.py")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC as ECA, FARM, NUM, N, TRAIN  # noqa: E402
ridf = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; IDX = {ridf(i): i for i in range(N)}
PMx = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); LINK = {r: (p, float(q)) for r, p, q in zip(PMx.rec, PMx.prev0, PMx.p0) if isinstance(p, str) and q >= .3 and p != r}
_all = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"), usecols=["rec", "lock"]); LOCK = set(_all[_all.lock].rec)
def a_ok(p, val): return p is not None and p in IDX and TRAIN[IDX[p]] and p not in LOCK and p not in val and np.isfinite(ECA[IDX[p], 23])
def extra(r, base, h, val):
    p, q = LINK.get(r, (None, 0.0))
    if a_ok(p, val): return np.array([ECA[IDX[p], 23] - np.nanmean(base[:h + 1]), q, 1.0])
    return np.array([0.0, q, 0.0])
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
DH5, DH6, used = {}, {}, []
def mk():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True)
for v in ("DIAG10", "EL1", "P2LOO"):
    for k, Gk in O[O.validator == v].groupby("fold"):
        val = set(Gk.rec); hide = np.zeros(len(TRK), bool)
        for r in val:
            f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTHER[f]) & (np.abs(tnum - n) <= 3))
        tr = ~hide; M5, M6 = {}, {}
        for h in HS:
            X6 = np.concatenate([TF[h], np.stack([extra(r, TR[r][0], h, val) for r in TRK])], 1)
            m = mk(); m.fit(TF[h][tr], TD[tr]); M5[h] = m; m = mk(); m.fit(X6[tr], TD[tr]); M6[h] = m
        for (sd, r), g in Gk.groupby(["seed", "rec"]):
            g = g.sort_values("hour"); base = g.sg.values
            if len(base) != 24: continue
            Q = np.stack([feat(INP[r], base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS]))
            E = np.stack([extra(r, base, h, val) for h in HS]); used.append((v, r, bool(E[0, 2])))
            d5 = {h: float(M5[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}; d6 = {h: float(M6[h].predict(np.concatenate([Q[j], E[j]])[None])[0]) for j, h in enumerate(HS)}
            DH5[(v, k, sd, r)] = np.array([d5[max(x for x in HS if x <= hh)] for hh in range(24)]); DH6[(v, k, sd, r)] = np.array([d6[max(x for x in HS if x <= hh)] for hh in range(24)])
        print(v, k, "done", flush=True)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
rows = []
for (v, k, sd), G in O.groupby(["validator", "fold", "seed"]):
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour")
        if (v, k, sd, r) not in DH5: continue
        rows.append(pd.DataFrame(dict(validator=v, seed=sd, rec=r, farm=g.farm.values, day=g.day.values, hour=g.hour.values, y=g.sub_ec.values, base=g.sg.values,
                                      hr5=g.sg.values + .5 * DH5[(v, k, sd, r)], hr6=g.sg.values + .5 * DH6[(v, k, sd, r)])))
B = pd.concat(rows, ignore_index=True); B["d"] = (B.hr6 - B.y) ** 2 - (B.hr5 - B.y) ** 2; B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
U = pd.DataFrame(used, columns=["val", "rec", "used"]).drop_duplicates(); print("records with yesterday EC used:", U.groupby("val").used.mean().round(3).to_dict())
res = []
for (v, sd), g in B.groupby(["validator", "seed"]):
    p2 = g.p2.values; f = lambda m, c: 100 * (rm((g[c] - g.y)[m]) / rm((g.hr5 - g.y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE_p2=float(g.d[p2].sum()), p2=f(p2, "hr6"), all=f(np.ones(len(g), bool), "hr6"), p2_normal=f(p2 & ~g.hi.values, "hr6"), p2_high=f(p2 & g.hi.values, "hr6"),
                    hr6_vs_base_all=100 * (rm(g.hr6 - g.y) / rm(g.base - g.y) - 1), hr5_vs_base_all=100 * (rm(g.hr5 - g.y) / rm(g.base - g.y) - 1), X0_p2=float(g.d[p2 & ~g.rec.isin(LEDGER).values].sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(v, p2only):
    g = B[(B.validator == v) & (B.p2 if p2only else True)]
    s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20261229).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot("P2LOO", True)
RD = B[B.validator.isin(["P2LOO", "EL1"]) & B.p2].groupby(["validator", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum()); c1b = pP2 < .025 or nh == 0
_, pb = boot("DIAG10", False); c3 = not (B[B.validator == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
print(f"\n(1) main: cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP2:.4f} | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
print(f"(3) DIAG10 guard: mean dSSE {B[B.validator == 'DIAG10'].groupby('seed').d.sum().mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("record dSSE (all validators, seed mean) best/worst:", B.groupby("rec").d.sum().div(3).sort_values().round(3).head(5).to_dict(), B.groupby("rec").d.sum().div(3).sort_values().round(3).tail(5).to_dict())
print("HR6:", "ADOPT" if (c1 and c1b and c3) else "FAIL")
B.to_csv(os.path.join(HERE, "hr6_rows_v1.csv.gz"), index=False)
