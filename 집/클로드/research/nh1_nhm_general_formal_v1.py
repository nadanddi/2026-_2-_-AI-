# -*- coding: utf-8 -*-
"""NH1 (2026-10-14 집 클로드; 사용자: 'NHM 구성으로 정식 시험 진행해봐').  사전 고정(커밋 후 실행).  EC.
GEN1-EC(서술, 결과 이미 봄)에서 NHM(일반 방법만: R3+TabPFN+수축 → HR5 하루수준 보정 .5 → 0.9·NH + 0.1·MT1)이
구조판 S(SG2 포함)보다 EL1·P2LOO 에서 나았음.  같은 자료로 판정하면 사후 선택이므로, 독립 자료로 다시 판정:
  기본 예측 = SZ2(JOB001, 연구실 GPU) 체크포인트 — 새 R3 시드 1357·2468·3579, 새 PFN 문맥 21~24, 새 폴드 배치(DIAG10·EL1·P2LOO).
  ec_rv1(GEN1 이 쓴 예측) 은 보정기 학습 기준(DIAG10 nosg 시드 평균, HR5 학습 행)으로만 씀 — 평가 행에는 안 씀.
안 (두 개, 본페로니 k = 2 → P 기준 .0125):
  A1 = NHM(base = pre_SEA_s)  vs  S = fin_SEA_s         (계절 지표 포함 일반판 → 구조 모듈 SG2 대체)
  A2 = NHM(base = pre_NOS_s)  vs  S = fin_SEA_s         (계절 지표까지 뺀 완전 일반판)
  NHM(base) = 0.9·(base + .5·HR5(base)) + 0.1·MT1  — 상수 .5·.1 은 GEN1 과 같게 고정(새로 고르지 않음).
  HR5: 시각 h' ∈ {0,3,…,21} TabPFN v2(n_est 4, rs 0), 특징 = 0..h' 누적/밤/낮 평균(14 입력 + base),
       정의상 같은 열은 뺌(제출 24 패키지와 같게: 밤 평균은 h'>7, 낮 평균은 h'≥9 일 때만), 목표 = 하루 평균 잔차.
  MT1: 6.894 의 B(GRU 2층 128 + 보조 머리 + mix), 내부 시드 1·2.
  학습 제외(HR5·MT1 모두, SZ2 와 같음): 검증 날 기준 같은 농장 ±1, 다른 농장 −3..+3.  (잠금일은 HR5 학습 행 A 에서 이미 빠짐.)
  24시간이 다 있는 기록만 평가(빠지는 행 수 보고). S 도 같은 행으로 비교.
판정(EC 분야 배정 규칙, 안마다):
  (1) P2LOO·EL1 3시드 모두 2차 dSSE(안 − S) < 0  AND  (P2LOO 시드 평균 농장 x 5일 블록 부트스트랩 P(악화) < .0125 OR 나빠진 2차 기록-칸 0)
  (3) DIAG10 손해 감시: 전체 행 시드 평균 dSSE > 0 AND '안 나음' 비율 < .025 이면 FAIL.
  둘 다 통과 → ADOPT.  A1·A2 둘 다 통과하면 A2(더 일반) 를 우선 후보로 보고.
서술: 1차/2차, 보통/고EC, 장부 기록 제외 dSSE(X0, 미결 규칙 — 판정에 안 씀), S+HR5(제출 24형) 대비.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u nh1_nhm_general_formal_v1.py {prep|judge}
  prep : MT1 예측(폴드 배치만 필요, 기본 예측 불필요) 캐시 → local/nh1_mt1_v1.pkl
  judge: SZ2 체크포인트 66개가 다 있어야 실행(없으면 중단).
"""
import os, sys, glob, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "prep"
src = open(os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr5_tabpfn_dayresid_causal_formal_v1.py")
exec(src[:src.index("TR = {}")])
src = open(os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "mt1_mtl_aux_future_inputs_v1.py")
exec(src[:src.index("O = pd.read_csv")])
__file__ = os.path.join(HERE, "nh1_nhm_general_formal_v1.py")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
CK = os.path.join(env.LOCAL, "sz2_ckpt"); MTF = os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"); SEEDS = (1357, 2468, 3579); N_FOLDS = 62   # SZ2 folds() 실제 개수(처음 66 은 어림 오류, 판정 기준 불변)
LAM, WM, PB = .5, .1, .0125
def keep(h): return [0] + ([1] if h > 7 else []) + ([2] if h >= 9 else [])
def featk(inp, base, h):
    f = feat(inp, base, h); d = len(f) // 3; return np.concatenate([f[j * d:(j + 1) * d] for j in keep(h)])
def fold_sets():
    # SZ2 의 폴드 배치를 SZ2 함수로 그대로 재현 (ct2.prep 필요) — 무거우므로 체크포인트가 있으면 거기서 읽음
    if len(glob.glob(os.path.join(CK, "*.csv"))) >= N_FOLDS:
        out = {}
        for f in sorted(glob.glob(os.path.join(CK, "*.csv"))):
            g = pd.read_csv(f, usecols=["farm", "day", "validator", "validation_fold"]); out[(g.validator.iloc[0], int(g.validation_fold.iloc[0]))] = sorted({(a, int(b)) for a, b in zip(g.farm, g.day)})
        return out
    import importlib.util
    sys.argv = ["x", "1"]; spec = importlib.util.spec_from_file_location("sz2", os.path.join(HERE, "sz2_season_index_real_pipeline_v1.py"))
    sz2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sz2)
    raw, full, lab, lock, fds = sz2.ct2.prep(); out = {}
    for name, i, vd in sz2.folds(lab, fds):
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}; out[(name, i)] = sorted(vd)
    return out
def hidden(vd, recs):
    return {r for r in recs if any((r[:3] == f and abs(int(r[4:]) - d) <= 1) or (r[:3] == OTHER[f] and abs(int(r[4:]) - d) <= 3) for f, d in vd)}
rid = lambda f, d: "%s_%03d" % (f, d)
if MODE == "prep":
    FS = fold_sets(); print("folds", len(FS), flush=True); MP = {}
    if os.path.exists(MTF): MP = pickle.load(open(MTF, "rb"))
    for key, vd in FS.items():
        if key in MP: continue
        te = [rid(f, d) for f, d in vd if rid(f, d) in idx]; hide = hidden(vd, LABREC) | set(te); tr = [r for r in LABREC if r not in hide]
        p, _ = fit_predict(tr, te, "B"); MP[key] = {r: p[j] for j, r in enumerate(te)}
        pickle.dump(MP, open(MTF, "wb")); print("MT1", key, "done", flush=True)
    print("prep done", len(MP)); sys.exit(0)
# ---- judge ----
assert len(glob.glob(os.path.join(CK, "*.csv"))) >= N_FOLDS, "SZ2 checkpoints incomplete"
MP = pickle.load(open(MTF, "rb"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
NB = O[O.validator == "DIAG10"].groupby("row_id").nosg.mean(); TR = {}
for r, g in A.groupby("rec"):
    rid_ = [f"{r}_{h:02d}" for h in g.hour]; b = NB.reindex(rid_).values
    if len(g) == 24 and np.isfinite(b).all() and np.isfinite(g.y.values).sum() >= 12 and r in INP:
        TR[r] = (b, float(np.nanmean(g.y.values) - b.mean()))
TRK = sorted(TR); TF = {h: np.stack([featk(INP[r], TR[r][0], h) for r in TRK]) for h in HS}; TD = np.array([TR[r][1] for r in TRK])
for h in HS: TF[h] = np.where(np.isfinite(TF[h]), TF[h], np.nanmean(TF[h], 0))
print("HR5 training records", len(TRK), flush=True)
res, skipped = [], 0
for f in sorted(glob.glob(os.path.join(CK, "*.csv"))):
    G = pd.read_csv(f); v, k = G.validator.iloc[0], int(G.validation_fold.iloc[0]); vd = sorted({(a, int(b)) for a, b in zip(G.farm, G.day)})
    G["rec"] = [rid(a, b) for a, b in zip(G.farm, G.day)]; hide = hidden(vd, TRK); msk = np.array([r not in hide for r in TRK])
    models = {}
    for h in HS:
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True); m.fit(TF[h][msk], TD[msk]); models[h] = m
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour")
        if len(g) != 24 or r not in INP or r not in MP.get((v, k), {}): skipped += len(g); continue
        mt = MP[(v, k)][r][g.hour.values]; out = dict(validator=v, fold=k, rec=r, farm=g.farm.values, day=g.day.values, hour=g.hour.values, y=g.sub_ec.values)
        for s in SEEDS:
            for arm, col in (("A1", f"pre_SEA_{s}"), ("A2", f"pre_NOS_{s}")):
                base = g[col].values; dh = {}
                for h in HS:
                    q = featk(INP[r], base, h)[None]; q = np.where(np.isfinite(q), q, TF[h].mean(0)); dh[h] = float(models[h].predict(q)[0])
                d24 = np.array([dh[max(x for x in HS if x <= hh)] for hh in g.hour.values])
                out[f"{arm}_{s}"] = (1 - WM) * (base + LAM * d24) + WM * mt
                if arm == "A1": out[f"SH_{s}"] = g[f"fin_SEA_{s}"].values + LAM * d24
            out[f"S_{s}"] = g[f"fin_SEA_{s}"].values
        res.append(pd.DataFrame(out))
    print(v, k, "done", flush=True)
B = pd.concat(res, ignore_index=True); B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "fold", "rec"]).y.transform("mean") >= 1.0
B.to_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"), index=False); print("rows", len(B), "skipped rows (not 24h)", skipped)
pd.set_option("display.width", 220); VERD = {}
for arm in ("A1", "A2"):
    rows, blocks, recd = [], {}, []
    for v in ("DIAG10", "EL1", "P2LOO"):
        g = B[B.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values
        for s in SEEDS:
            a, n = g[f"S_{s}"].values, g[f"{arm}_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
            fx = lambda m: 100 * (rm((n - y)[m]) / rm((a - y)[m]) - 1) if m.any() else np.nan
            led = ~g.rec.isin(LEDGER).values
            rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), all=fx(np.ones(len(y), bool)), p1=fx(~p2), p2=fx(p2), p2_normal=fx(p2 & ~g.hi.values), p2_high=fx(p2 & g.hi.values),
                             X0_dSSE_p2=float(d[p2 & led].sum()), vsSH_p2=100 * (rm((n - y)[p2]) / rm((g[f"SH_{s}"].values - y)[p2]) - 1) if p2.any() else np.nan))
            blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
            for r_, dd in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd))
    S = pd.DataFrame(rows); print(f"\n=== {arm} vs S (rel RMSE %, negative = {arm} better) ==="); print(S.round(3).to_string(index=False))
    def boot(key_):
        x = pd.concat(blocks[key_], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20261226).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot(("P2LOO", "p2"))
    RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum()); c1b = pP2 < PB or nh == 0
    dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
    print(f"(1) main: cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP2:.4f} (< {PB}) | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
    print(f"(3) DIAG10 harm guard: all-rows mean dSSE {dg.dSSE_all.mean():+.3f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
    print("X0 (ledger excluded, descriptive): P2LOO+EL1 pass-2 dSSE seed-sum", round(float(main.X0_dSSE_p2.sum()), 3))
    print("P2LOO pass-2 records worst (seed-mean dSSE):", RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values().round(3).tail(6).to_dict())
    VERD[arm] = c1 and c1b and c3; print(f"NH1 {arm}:", "ADOPT" if VERD[arm] else "FAIL")
print("\nNH1 summary:", VERD, "-> preferred:", "A2" if VERD["A2"] else ("A1" if VERD["A1"] else "none"))
