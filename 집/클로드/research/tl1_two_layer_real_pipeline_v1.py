# -*- coding: utf-8 -*-
"""TL1 (2026-10-14 집 클로드; 사용자: '실제 기본 모델로 새 배치에서 확인해봐').  사전 고정(커밋 후 실행).  EC.
P2T2/P2T2B(6.848·6.849, 대리 ET): '2차 + 같은 시기 1차' 로 학습하면 보통 날 −14~−17%, 고EC +26%; 관문(전체 모델 누적평균 >= 1.0)으로 두 층 → −3.4%(사후 관찰).
실제 파이프라인(CT2 와 같음 = submission_14 형): pre = clip(shrink(.8 R3 + .2 PFN)), R3 = .6 ET + .3 LGB + .1 MLP(DC4 계절 + DP1), PFN = TabPFN v2(문맥 2000행 x 4),
  fin = SG2 보정(2차 행) 후 clip.
층:  REF = 전체 학습 자료(지금과 같음);  NEAR = 같은 파이프라인을 '2차 날 + 학습 달력(DMT3e) 으로 어떤 2차 날과 같은 구간·날짜 차 <= 2 인 1차 날' 행만으로 학습
  (R3·PFN 문맥 모두 그 행에서; 계절 지표는 같은 폴드 전체 학습 날로 — P2T1 수정과 같음; SG2 참조는 REF 와 같음).
후보 TL = 시각 h 에서 관문 g_h = REF pre 의 0..h 누적평균 >= 1.0 이면 REF fin, 아니면 NEAR fin  (문턱 1.0 = 기존 기준, 결과 보고 고르지 않음).
새 시드 2121·3434·5656, 새 PFN 문맥 13~16.  학습 제외: 같은 농장 ±1, 다른 농장 −3..+3, 잠금 ±1(CT2 와 같음).
배치: (1) 새 배치 EL4o2 = 농장별 2차 날을 번호순으로 3번째부터 4개씩 묶고 앞 2개는 마지막 묶음에 붙임(처음 쓰는 배치);  (2) P2LOO(이 구상엔 처음).
판정(k = 1, 미리 정함): 2차 행에서 (a) 3시드 x {EL4o2, P2LOO} 모두 TL < REF;  (b) EL4o2 시드 평균 (농장, 날) 묶음 부트스트랩 20,000 에서 'TL 이 낫지 않은' 비율 < .025.
서술: 보통/고EC 날, 관문 경로 비율, NEAR 단독.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u tl1_two_layer_real_pipeline_v1.py {cv|sum}
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "1"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
wt0, dc4, p3, core, sg2, dp1 = ct2.wt0, ct2.dc4, ct2.p3, ct2.core, ct2.sg2, ct2.dp1
SEEDS = (2121, 3434, 5656); CTX = (13, 14, 15, 16); W = (.6, .3, .1); OTHER = {"F13": "F47", "F47": "F13"}
CK = os.path.join(env.LOCAL, "tl1_ckpt")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s_, d_) for r, s_, d_ in zip(CAL.rec, CAL.seg, CAL.day)}
key = lambda f, d: f"{f}_{int(d):03d}"
def folds(lab):
    out = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()); ds2 = ds[2:]; grp = [ds2[k:k + 4] for k in range(0, len(ds2), 4)]; grp[-1] = list(grp[-1]) + list(ds[:2])
        for g in grp: out.append(("EL4o2", len(out), {(f, int(d)) for d in g}))
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()): out.append(("P2LOO", 500 + len(out), {(f, int(d))}))
    return out
def run_cv():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, fds = ct2.prep(); wv = dc4.weather_vectors(full); cols, PF = ct2.feature_sets(); FS, BS = cols["REF"]
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    p2dates = [DAY[key(f, d)] for f, d in labset if d >= 179 and key(f, d) in DAY]
    near1 = {k for k, (s_, d_) in DAY.items() if int(k[4:]) < 179 and any(s_ == s2 and abs(d_ - d2) <= 2 for s2, d2 in p2dates)}
    print("NEAR pass-1 days:", len(near1), flush=True)
    for name, i, vd in folds(lab):
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
        nm = np.array([(d >= 179) or (key(f, d) in near1) for f, d in zip(tr.farm, tr.day)]); trN = tr[nm].copy()
        lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
        Xva = va[PF].to_numpy(np.float32); vaR = va.reset_index(drop=True)
        G = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True); G["validator"], G["validation_fold"] = name, i
        ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        for lay, T in (("REF", tr), ("NEAR", trN)):
            Xt, yt = T[PF].to_numpy(np.float32), T.sub_ec.to_numpy(float)
            pf = np.mean([ct2.pfn(Xt[np.random.default_rng(c).choice(len(T), min(2000, len(T)), replace=False)], yt[np.random.default_rng(c).choice(len(T), min(2000, len(T)), replace=False)], Xva, c) for c in CTX], axis=0)
            for s in SEEDS:
                e, l, m = wt0.members(T, va, s, FS, BS); r3 = W[0] * e + W[1] * l + W[2] * m
                G["pre_%s_%d" % (lay, s)] = np.clip(ct2.shrink(.8 * r3 + .2 * pf, vaR), lo, hi)
                G["fin_%s_%d" % (lay, s)] = np.clip(sg2.correction(G, "pre_%s_%d" % (lay, s), vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        G["ntrain_near_days"] = trN[["farm", "day"]].drop_duplicates().shape[0]
        G.to_csv(path, index=False); print("%s/%d done (NEAR days %d)" % (name, i, G.ntrain_near_days.iloc[0]), flush=True)
def summarize():
    import glob
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CK, "*.csv"))], ignore_index=True)
    D = D[D.day >= 179].sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    D["hi"] = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1.0; rows = []
    for s in SEEDS:
        D[f"cm_{s}"] = D.groupby(["validator", "validation_fold", "farm", "day"])[f"pre_REF_{s}"].transform(lambda x: x.expanding().mean())
        D[f"tl_{s}"] = np.where(D[f"cm_{s}"] >= 1.0, D[f"fin_REF_{s}"], D[f"fin_NEAR_{s}"])
        for v, g in D.groupby("validator"):
            y = g.sub_ec.values; f = lambda c, m=None: rm((g[c].values - y)[m]) if m is not None else rm(g[c].values - y)
            rows.append(dict(val=v, seed=s, REF=f(f"fin_REF_{s}"), NEAR=f(f"fin_NEAR_{s}"), TL=f(f"tl_{s}"), REF_normal=f(f"fin_REF_{s}", ~g.hi.values), TL_normal=f(f"tl_{s}", ~g.hi.values),
                             REF_high=f(f"fin_REF_{s}", g.hi.values), TL_high=f(f"tl_{s}", g.hi.values), routed_high=float((g[f"cm_{s}"] >= 1.0).mean())))
    S = pd.DataFrame(rows); S["TL_pct"] = 100 * (S.TL / S.REF - 1); pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
    a = bool((S.TL < S.REF).all())
    g = D[D.validator == "EL4o2"]; keys, inv = np.unique((g.farm + "_" + g.day.astype(str)).values, return_inverse=True)
    dd = np.mean([np.bincount(inv, weights=(g[f"tl_{s}"] - g.sub_ec) ** 2 - (g[f"fin_REF_{s}"] - g.sub_ec) ** 2, minlength=len(keys)) for s in SEEDS], axis=0)
    ix = np.random.default_rng(20261207).integers(0, len(dd), (20000, len(dd))); P = float((dd[ix].sum(1) >= 0).mean())
    print(f"\n(a) all 6 cells TL < REF: {a} ({int((S.TL < S.REF).sum())}/6) | (b) EL4o2 day-bootstrap share(TL not better) {P:.4f} | days better {int((dd < 0).sum())}/{len(dd)}")
    print("TL1:", "ADOPT" if (a and P < .025) else "FAIL")
if __name__ == "__main__":
    run_cv() if MODE == "cv" else summarize()
