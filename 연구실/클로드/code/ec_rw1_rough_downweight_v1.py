# -*- coding: utf-8 -*-
"""RW1 — CO₂ 거친 날(30일) 학습 가중치 0.2 + 거친 날 뺀 검증 점수 (계획 v2 고정, 2026-10-08 연구실 클로드)
계획: 연구실/클로드/문서/RW1_거친날_가중하향_계획_v2.md
기준·후보 모두 이 프로세스에서 새로 학습(WT0 구성: ET FULL+season+DP1, LGB-tweedie BASE+season+DP1, MLP BASE+season+DP1, 시드 47/1414/6464).
폴드: DIAG10(10)·A(5)·B(5)·EL1(2차 5일 묶음). 학습 제외 ±1일·잠금 날은 WT0 와 같음.
실행: cd 연구실/클로드 && PYTHONPATH="" python -u code/ec_rw1_rough_downweight_v1.py  (체크포인트 local/rw1_ckpt)
"""
import os, sys, hashlib, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
HR = os.path.join(HERE, "..", "..", "..", u"집", u"클로드", "research"); sys.path.insert(0, HR)
import env  # noqa
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HR, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = (47, 1414, 6464); W_R = 0.2
OUT = os.path.join(HERE, ".."); CK = os.path.join(OUT, "local", "rw1_ckpt")
RF = os.path.join(OUT, "results", "rw1_rough_days_v1.csv")
assert hashlib.sha256(open(RF, "rb").read()).hexdigest()[:16] == "2273ec3f93970cdb"
ROUGH = set(map(tuple, pd.read_csv(RF)[["farm", "day"]].itertuples(index=False, name=None)))


def members(tr, va, s, FS, BS, w):
    et = core.et(s); et.fit(tr[FS], tr.sub_ec.to_numpy(), **({} if w is None else {"extratreesregressor__sample_weight": w}))
    lg = core.lg(s, "tweedie"); lg.fit(tr[BS], tr.sub_ec.to_numpy(), **({} if w is None else {"sample_weight": w}))
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        core.MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                                          early_stopping=True, n_iter_no_change=25, validation_fraction=.12, random_state=s))
    mlp.fit(tr[BS], tr.sub_ec.to_numpy(), **({} if w is None else {"mlpregressor__sample_weight": w}))
    return et.predict(va[FS]), lg.predict(va[BS]), mlp.predict(va[BS])


def run_folds():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    folds = [x for x in fds if x[0] in ("DIAG10", "A", "B")]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    checked = False
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        w = np.where([(f, int(d)) in ROUGH for f, d in zip(tr.farm, tr.day)], W_R, 1.0)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max(); frame["n_rough_train_rows"] = int((w < 1).sum())
        for s in SEEDS:
            for tag, ww in (("b", None), ("c", w)):
                e, l, m = members(tr, va, s, FS, BS, ww)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (tag, nm, s)] = core.shrink(v, va)
            if not checked:
                d = {nm: float(np.max(np.abs(frame["b_%s_%d" % (nm, s)] - frame["c_%s_%d" % (nm, s)]))) for nm in ("et", "lgb", "mlp")}
                print("가중치 적용 확인(첫 폴드, 시드 %d) 최대 |기준−후보|:" % s, d, "거친 학습 행", int((w < 1).sum()), flush=True)
                assert all(v > 0 for v in d.values()), "가중치가 어떤 구성원에 안 들어감"
                checked = True
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def evaluate():
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(OUT, "local", "ec_rw1_all_v1.csv"), index=False)
    O["rough"] = [(f, int(d)) in ROUGH for f, d in zip(O.farm, O.day)]
    C = pd.read_csv(os.path.join(OUT, "results", "ec_co2r_days_v1.csv"))
    sealed = {(f, int(d)): bool(s) for f, d, s in zip(C.farm, C.day, C.sealed)}
    O["sealed"] = [sealed.get((f, int(d)), False) for f, d in zip(O.farm, O.day)]
    assert all((f, int(d)) in sealed for f, d in zip(O.farm, O.day)), "밀폐 표에 없는 날"  # 중간 비평 2
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    def pred(G, tag, s):
        return np.clip(.6 * G["%s_et_%d" % (tag, s)] + .3 * G["%s_lgb_%d" % (tag, s)] + .1 * G["%s_mlp_%d" % (tag, s)], G.lo, G.hi)
    def pboot(T, seed=20261008):
        T = T.copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
        bm = np.mean([pred(T, "b", s) for s in SEEDS], axis=0); cm = np.mean([pred(T, "c", s) for s in SEEDS], axis=0)
        dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values; idx = np.random.default_rng(seed).integers(0, len(sm), (20000, len(sm)))
        return r(bm - T.sub_ec), r(cm - T.sub_ec), float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("주: 시드평균 = 세 시드 예측을 평균한 뒤의 RMSE(WT0 관례). EC 3분위는 모든 검증기 날을 합쳐 계산(서술용).")
    print("\n=== 판정 (R3 수준) ===")
    ok_a = True; ok_c = True
    for v in ("DIAG10", "A", "B", "EL1"):
        G = O[O.validator == v]
        for lbl, H in (("거친 날 제외", G[~G.rough]), ("전체", G)):
            cells = []
            for s in SEEDS:
                a, b = r(pred(H, "b", s) - H.sub_ec), r(pred(H, "c", s) - H.sub_ec)
                if v != "EL1" and lbl == "거친 날 제외": ok_a &= b < a
                cells.append("s%d %.4f→%.4f (%+.2f%%)" % (s, a, b, 100 * (b / a - 1)))
            bm, cm, p = pboot(H)
            if v != "EL1" and lbl == "전체": ok_c &= cm <= bm
            print(f"  {v:6s} {lbl:7s} 날 {H.groupby(['farm','day']).ngroups:3d} | " + "  ".join(cells) + f" | 시드평균 {bm:.4f}→{cm:.4f} ({100*(cm/bm-1):+.2f}%) P(worse) {p:.4f}")
    D = O[(O.validator == "DIAG10") & ~O.rough]; _, _, pb = pboot(D)
    guards = []
    for v in ("DIAG10", "EL1"):
        G = O[(O.validator == v) & ~O.rough]
        for lbl, H in (("2차", G[G.day >= 179]), ("밀폐", O[(O.validator == v) & O.sealed])):
            if len(H) == 0: continue
            bm, cm, _ = pboot(H); ch = 100 * (cm / bm - 1); guards.append((v, lbl, ch))
            print(f"  안전장치 {v} {lbl}: {bm:.4f}→{cm:.4f} ({ch:+.2f}%)")
        Hs = O[(O.validator == v) & O.sealed & ~O.rough]  # 중간 비평 1: 서술만, 판정 불변
        if len(Hs): bm, cm, _ = pboot(Hs); print(f"  (서술) {v} 밀폐·거친 날 제외: {bm:.4f}→{cm:.4f} ({100*(cm/bm-1):+.2f}%)")
    hold = any(ch >= 2 for _, _, ch in guards)
    passed = ok_a and pb < .025 and ok_c
    print(f"\nRW1 판정: (a) 거친 날 제외 전 시드×DIAG10·A·B 개선 {ok_a}, (b) DIAG10 P(worse) {pb:.4f} {'< .025' if pb < .025 else '≥ .025'}, (c) 전체 행 시드평균 악화 없음 {ok_c} → {'PASS' if passed else 'FAIL'}{' (안전장치 보류)' if passed and hold else ''}")
    # PFN 혼합 (서술)
    WT = pd.read_csv(os.path.join(OUT, "local", "drive_copy", "ec3_WT1_all.csv"))[["row_id", "validator", "pfn"]]
    for v in ("DIAG10", "EL1"):
        G = O[O.validator == v].merge(WT[WT.validator == v], on=["row_id", "validator"], how="left")
        if G.pfn.isna().any(): print(f"  PFN 혼합 {v}: PFN 결측 {G.pfn.isna().sum()}행 — 생략"); continue
        for lbl, H in (("거친 날 제외", G[~G.rough]), ("전체", G)):
            bm = np.mean([np.clip(.8 * pred(H, "b", s) + .2 * H.pfn, H.lo, H.hi) for s in SEEDS], axis=0)
            cm = np.mean([np.clip(.8 * pred(H, "c", s) + .2 * H.pfn, H.lo, H.hi) for s in SEEDS], axis=0)
            print(f"  PFN 혼합(0.8R3+0.2PFN, 서술) {v} {lbl}: {r(bm - H.sub_ec):.4f}→{r(cm - H.sub_ec):.4f}")
    # 부풀림 진단 (기준 모델)
    print("\n=== 부풀림 진단 (기준 R3 시드평균) ===")
    rng = np.random.default_rng(7)
    dm = O.groupby(["farm", "day"]).sub_ec.mean(); tert = pd.qcut(dm, 3, labels=False).to_dict()
    for v in ("DIAG10", "A", "B", "EL1"):
        G = O[O.validator == v].copy(); G["bp"] = np.mean([pred(G, "b", s) for s in SEEDS], axis=0)
        ds = G.groupby(["farm", "day"]).apply(lambda g: pd.Series({"sse": ((g.bp - g.sub_ec) ** 2).sum(), "n": len(g), "rough": g.rough.iloc[0], "sealed": g.sealed.iloc[0]}), include_groups=False)
        ds["tert"] = [tert[k] for k in ds.index]; ds["p2"] = [k[1] >= 179 for k in ds.index]
        nr = int(ds.rough.sum()); allr = np.sqrt(ds.sse.sum() / ds.n.sum()); exr = np.sqrt(ds[~ds.rough].sse.sum() / ds[~ds.rough].n.sum())
        if nr < 3:
            print(f"  {v}: 거친 날 {nr}일 — 판단 불가 (전체 {allr:.4f}, 제외 {exr:.4f})"); continue
        rd = ds[ds.rough]; pool = ds[~ds.rough]; sims_a = []
        for _ in range(500):
            drop = []
            for k, row in rd.iterrows():
                cand = pool[(pool.index.get_level_values(0) == k[0]) & (pool.sealed == row.sealed) & (pool.tert == row.tert) & (pool.p2 == row.p2)]
                cand = cand[~cand.index.isin(drop)]
                if len(cand) == 0: cand = pool[(pool.index.get_level_values(0) == k[0]) & ~pool.index.isin(drop)]
                drop.append(cand.index[rng.integers(len(cand))])
            keep = ds[~ds.index.isin(drop)]; sims_a.append(np.sqrt(keep.sse.sum() / keep.n.sum()))
        # 블록 짝: 거친 날 연속 구간 길이별로 같은 온실의 연속 비거친 검증 날 블록
        runs = []
        for f in ("F13", "F47"):
            dd = sorted(d for (ff, d) in rd.index if ff == f); cur = []
            for d in dd:
                if cur and d - cur[-1] > 1: runs.append((f, len(cur))); cur = []
                cur.append(d)
            if cur: runs.append((f, len(cur)))
        sims_b = []; nskip = 0
        for _ in range(500):
            drop = set()
            for f, L in runs:
                pd_ = sorted(d for (ff, d) in pool.index if ff == f and (ff, d) not in drop)
                starts = [j for j in range(len(pd_) - L + 1) if pd_[j + L - 1] - pd_[j] <= 2 * (L - 1) + 1]
                if not starts: nskip += 1; continue
                j = starts[rng.integers(len(starts))]; drop |= {(f, d) for d in pd_[j:j + L]}
            keep = ds[~ds.index.isin(list(drop))]; sims_b.append(np.sqrt(keep.sse.sum() / keep.n.sum()))
        qa, qb = np.percentile(sims_a, 2.5), np.percentile(sims_b, 2.5)
        print(f"      (블록 짝: 연속 블록을 못 찾아 건너뛴 구간 수 합 {nskip}, 구간 {len(runs)}개 × 500회)")
        print(f"  {v}: 거친 {nr}일 | 전체 {allr:.4f} | 거친 날 제외 {exr:.4f} | 날 짝 대조 평균 {np.mean(sims_a):.4f} (2.5% {qa:.4f}) | 블록 짝 대조 평균 {np.mean(sims_b):.4f} (2.5% {qb:.4f}) → {'부풀림' if exr < qa and exr < qb else '구별 안 됨'}")
        print(f"      거친 날 RMSE {np.sqrt(rd.sse.sum()/rd.n.sum()):.4f} vs 비거친 {exr:.4f}; 거친 날 SSE 몫 {rd.sse.sum()/ds.sse.sum():.2f} (날 비율 {nr/len(ds):.2f})")


if __name__ == "__main__":
    run_folds()
    evaluate()
