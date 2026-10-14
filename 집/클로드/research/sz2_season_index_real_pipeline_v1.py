# -*- coding: utf-8 -*-
"""SZ2 (2026-10-14 집 클로드; 사용자: '실제 파이프라인으로 계절 지표 확인 진행해봐').  사전 고정(커밋 후 실행).  EC.
SZ1(6.897, 대리 ET): 계절 지표(번호 순서·날씨 재사용 구조에 의존)를 빼면 전체 +10% 악화, 2차 −10% 개선.
여기선 실제 파이프라인(CT2/TL1 과 같음): pre = clip(shrink(.8 R3 + .2 PFN)), R3 = .6 ET + .3 LGB + .1 MLP(DP1 운영 특징), PFN = TabPFN v2 문맥 2000행 x 4,
  fin = SG2 보정(2차 행) 후 clip.
안: SEA = 지금(R3·PFN 입력에 season 포함) vs NOS = R3 의 FS·BS 와 PFN 입력에서 season 만 제거(나머지 동일, SG2 동일).
새 시드 R3 1357·2468·3579, PFN 문맥 21~24.  학습 제외: 같은 농장 ±1, 다른 농장 −3..+3, 잠금 ±1(CT2 와 같음).
검증기: DIAG10(fds 중 2차 포함 폴드), EL1(농장별 2차 날 5개 묶음), P2LOO(2차 날 하나씩) — ct2 stage 1·2 와 같은 배치.
판정(EC 분야 배정 규칙, 링커 안 씀, k = 1): (1) P2LOO·EL1 3시드 모두 2차 dSSE(NOS − SEA) < 0 AND (P2LOO 시드 평균 농장 x 5일 블록 P(악화) < .025 OR 나빠진 2차 기록 0);
  (3) DIAG10 손해 감시: 전체 행 시드 평균 악화 AND '나음' 비율 < .025 이면 FAIL.
서술: 전체 행, 1차 행(DIAG10), 보통/고EC, SG2 전(pre) 비교.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u sz2_season_index_real_pipeline_v1.py {cv|sum}
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json, glob
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "1"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
wt0, dc4, sg2 = ct2.wt0, ct2.dc4, ct2.sg2
SEEDS = (1357, 2468, 3579); CTX = (21, 22, 23, 24); W = (.6, .3, .1); OTHER = {"F13": "F47", "F47": "F13"}
CK = os.path.join(env.LOCAL, "sz2_ckpt"); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def folds(lab, fds):
    out = [x for x in fds if x[0] == "DIAG10" and any(d >= 179 for _, d in x[2])]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5): out.append(("EL1", 300 + len(out), {(f, int(d)) for d in ds[k:k + 5]}))
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()): out.append(("P2LOO", 300 + len(out), {(f, int(d))}))
    return out
def run_cv():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, fds = ct2.prep(); wv = dc4.weather_vectors(full); cols, PF = ct2.feature_sets(); FS, BS = cols["REF"]
    ARMS = {"SEA": (FS, BS, PF), "NOS": ([c for c in FS if c != "season"], [c for c in BS if c != "season"], [c for c in PF if c != "season"])}
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    for name, i, vd in folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path): continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        lo, hi = tr.sub_ec.min(), tr.sub_ec.max(); vaR = va.reset_index(drop=True)
        G = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True); G["validator"], G["validation_fold"] = name, i
        ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        for arm, (fs, bs, pf) in ARMS.items():
            Xt, yt, Xv = tr[pf].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[pf].to_numpy(np.float32)
            pfp = np.mean([ct2.pfn(Xt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], yt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], Xv, c) for c in CTX], axis=0)
            for s in SEEDS:
                e, l, m = wt0.members(tr, va, s, fs, bs); r3 = W[0] * e + W[1] * l + W[2] * m
                G["pre_%s_%d" % (arm, s)] = np.clip(ct2.shrink(.8 * r3 + .2 * pfp, vaR), lo, hi)
                G["fin_%s_%d" % (arm, s)] = np.clip(sg2.correction(G, "pre_%s_%d" % (arm, s), vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        G.to_csv(path, index=False); print("%s/%d done" % (name, i), flush=True)
def summarize():
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CK, "*.csv"))], ignore_index=True)
    D["hi"] = D.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean") >= 1.0; D["p2"] = D.day >= 179
    rows, blocks, recd = [], {}, []
    for (v, s), g in [((v, s), D[D.validator == v]) for v in ("DIAG10", "EL1", "P2LOO") for s in SEEDS]:
        y = g.sub_ec.values; a, n = g[f"fin_SEA_{s}"].values, g[f"fin_NOS_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2; p2 = g.p2.values
        f = lambda m: 100 * (rm((n - y)[m]) / rm((a - y)[m]) - 1) if m.any() else np.nan
        rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), all=f(np.ones(len(y), bool)), p2=f(p2), p1=f(~p2), p2_normal=f(p2 & ~g.hi.values), p2_high=f(p2 & g.hi.values),
                         pre_p2=100 * (rm((g[f"pre_NOS_{s}"].values - y)[p2]) / rm((g[f"pre_SEA_{s}"].values - y)[p2]) - 1) if p2.any() else np.nan))
        key = (g.farm + "_" + (g.day // 5).astype(str)).values
        blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
        for r_, dd in pd.Series(d[p2]).groupby((g.farm + "_" + g.day.astype(str)).values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd))
    S = pd.DataFrame(rows); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
    def boot(key):
        x = pd.concat(blocks[key], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20261226).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot(("P2LOO", "p2"))
    RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum()); c1b = pP2 < .025 or nh == 0
    dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
    print(f"\n(1) main P2LOO+EL1: cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP2:.4f} | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
    print(f"(3) DIAG10 harm guard: all-rows mean dSSE {dg.dSSE_all.mean():+.3f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
    print("P2LOO pass-2 records (seed-mean dSSE NOS-SEA):", RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values().round(3).to_dict())
    print("SZ2:", "ADOPT (drop season index)" if (c1 and c1b and c3) else "FAIL (keep season index)")
if __name__ == "__main__":
    run_cv() if MODE == "cv" else summarize()
