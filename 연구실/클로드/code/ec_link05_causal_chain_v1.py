# -*- coding: utf-8 -*-
"""LINK5 — 시각별 인과 사슬 + 판별기 시드 묶음, SG2 위에 얹기 (실행 전 기준 고정) — 2026-10-06 연구실 클로드

규정(운영 중간 안내 2026-10-05 원문 2·3절, 공용/대회자료/공고/온라인미션_중간안내_20261005.pdf):
  시간 앞뒤 = row_id 의 상대 일차(기록 번호)와 시각. 평가 행 (기록 t, 시각 h) 은 같은 온실의
  '기록 번호 < t 인 기록 전체' + '기록 t 의 0..h 시' 입력만. 학습 자료(train_X·train_y)는 시점 제한 없이 참조.
  평가 입력·예측값으로 학습/갱신 없음, 평가 전체 통계 조정 없음.
LINK3/4 와 다른 점: (1) 현재 기록의 특징을 0..c 시로 제한(c = 시각 묶음 0,1,3,6,12,23; h 시 행은 c ≤ h 중 최대),
  자정 점프는 c=0 이면 앞 기록 추세만으로 보정, 하루 서명은 두 기록 모두 같은 0..c 시 평균으로 비교.
  (2) 보정 기준 = 0..h 시 SG2 예측 누적평균 pm_h (하루 평균 대신; 뒤 시각 예측값 사용 안 함).
  (3) 판별기 = LightGBM 5개 시드 확률 평균(묶음). 묶음 3개(시드 집합 A/B/C)를 '시드' 칸으로 사용.
  (4) 정답 없는 앞 기록 q(평가형) 의 23시 값 = q 의 23시 판단 결과 level(q) + (모델23 − 모델하루평균)(q) — q 는 기록 번호상 과거.
규칙(고정): h 시 행에서 top-1 앞 기록 q, 묶음 확률 ≥ .5, v = q 의 23시 값,
  |v − pm_h| ≤ .30 이면 pred_h = sg_h + (v − pm_h), 아니면 sg_h.  (보호 .30 = SG2 와 같은 폭)
검증: SG2 저장 OOF (ec3_SG2_all.csv) 의 DIAG10, DIAG10y 2차 46일, SG2 시드 23/808/9090.
  폴드 k: 정답 가림 = 폴드 k 기록; 후보 L = 정답 있는 기록 − 폴드 k − (폴드 k ±1 기록) − 잠금 40일;
  U = 폴드 k 의 2차 기록 + 실제 평가 기록. 폴드 k 의 1차 기록·잠금일은 후보에서 제외.
  가림 기록의 모델 = SG2 시드평균(해당 검증기), 실제 평가 기록의 모델 = submission_14 EC 열.
[고정 기준] 두 검증기 각각: 묶음 3 × SG2 시드 3 = 9칸 모두 SG2 보다 낮은 2차 시간 RMSE,
  시드평균에서 두 온실 모두 개선, 온실×5기록 묶음 부트스트랩 2000 P(worse) < .025. 둘 다 충족해야 통과.
  진단 기준이며 제출 채택 판정 아님(채택은 EL1 등 추가 검증과 사용자 결정).
"""
import os, sys, hashlib, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
print("source sha256", hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16], flush=True)
CUTS = [0, 1, 3, 6, 12, 23]
ENS = {"A": [7, 101, 2024, 47, 1414], "B": [11, 202, 3030, 58, 1515], "C": [13, 303, 4040, 69, 1616]}
SGS = [23, 808, 9090]

TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
TR = TR.merge(Y[["row_id", "sub_ec"]], on="row_id", how="left"); TE["sub_ec"] = np.nan
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE], ignore_index=True)
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; IN = ["in_temp", "in_hum", "in_co2"]
ACT = ["act_vent", "act_side", "act_shade", "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog"]
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
LOCK = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]}

SG = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv"))
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"]
FOLD = {"DIAG10": DP.groupby(["farm", "day"]).validation_fold.first().to_dict()}
labdays = X[~X.is_test].groupby(["farm", "day"]).size().index
FOLD["DIAG10y"] = {(f, d): ((d + 2) // 6) % 10 for f, d in labdays if (f, d) not in LOCK}
S14 = pd.read_csv(os.path.join(env.SUBMIT, u"09회차_2026-10-06(팀)", "submission_14.csv")).set_index("row_id").sub_ec

REC = {}; truth = {}
for f in ("F13", "F47"):
    Z = X[X.farm == f]
    D = np.array(sorted(Z.day.unique())); n = len(D)
    piv = {v: Z.pivot(index="day", columns="hour", values=v).reindex(D).values for v in W + IN + ACT + ["sub_ec"]}
    isT = Z.groupby("day").is_test.first().reindex(D).values
    lab = ~isT
    sc = {v: np.nanstd(np.diff(piv[v][lab], axis=1)) for v in W + IN}
    REC[f] = dict(D=D, piv=piv, isT=isT, idx={d: i for i, d in enumerate(D)})
    # 정답 연결 (LINK1~4 와 같은 CH2 오라클, 정답 있는 기록끼리, 전체 하루 입력 — 학습 자료라 허용)
    def J(v):
        A23, A22, B0, B1 = piv[v][:, 23], piv[v][:, 22], piv[v][:, 0], piv[v][:, 1]
        return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
    hand = sum(wt * (J(v) / sc[v]) ** 2 for v, wt in zip(W, (1, 1, .3, 1))) + sum((J(v) / sc[v]) ** 2 for v in IN) / 4 \
        + sum(np.abs(piv[v][:, 0][None, :] - piv[v][:, 23][:, None]) for v in ("act_heating", "act_thermal", "act_circfan", "act_vent")) / 200
    li = np.where(lab)[0]; nl = len(li); jec = J("sub_ec")[np.ix_(li, li)]
    M = hand[np.ix_(li, li)] + (jec / .02) ** 2; M = np.where(np.isfinite(M), M, 1e6); np.fill_diagonal(M, 1e6)
    big = np.full((2 * nl, 2 * nl), 1e6); big[:nl, :nl] = M
    big[:nl, nl:] = np.where(np.eye(nl) == 1, 12.0, 1e6); big[nl:, :nl] = np.where(np.eye(nl) == 1, 12.0, 1e6); big[nl:, nl:] = 0
    r_, c_ = linear_sum_assignment(big)
    for i, j in zip(r_, c_):
        if i < nl and j < nl and abs(jec[i, j]) < .05:
            truth[(f, D[li[j]])] = D[li[i]]
    # 시각 묶음별 쌍 특징 행렬 [a, b] (b 는 0..c 시만)
    F = {}
    for c in CUTS:
        fe = {}
        for v in W + IN:
            A23, A22, B0 = piv[v][:, 23], piv[v][:, 22], piv[v][:, 0]
            if c == 0:
                j = (B0[None, :] - A23[:, None]) - (A23 - A22)[:, None]
            else:
                B1 = piv[v][:, 1]
                j = (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
            fe[("jw_" if v in W else "ji_") + v] = np.abs(j / sc[v])
        fe["datec"] = sum(wt * fe["jw_" + v] ** 2 for v, wt in zip(W, (1, 1, .3, 1)))
        for v in ACT:
            fe["ja_" + v] = np.abs(piv[v][:, 0][None, :] - piv[v][:, 23][:, None])
            with np.errstate(all="ignore"):
                mc = np.nanmean(piv[v][:, :c + 1], axis=1)
            fe["sd_" + v] = np.abs(mc[:, None] - mc[None, :])
        with np.errstate(all="ignore"):
            nt = np.nanmean(piv["in_temp"][:, :min(c, 5) + 1], axis=1)
        fe["sd_nightT"] = np.abs(nt[:, None] - nt[None, :])
        fe["hand"] = fe["datec"] + sum(fe["ji_" + v] ** 2 for v in IN) / 4 + sum(fe["ja_" + v] for v in ("act_heating", "act_thermal", "act_circfan", "act_vent")) / 200
        fe["gap"] = D[None, :] - D[:, None] + 0.0
        fe["late_a"] = np.repeat((D >= 179)[:, None], n, 1) + 0.0
        fe["late_b"] = np.repeat((D >= 179)[None, :], n, 0) + 0.0
        fe["cut"] = np.full((n, n), float(c))
        F[c] = fe
    REC[f]["F"] = F
FEAT = list(REC["F13"]["F"][0].keys())
print("정답 연결 %d, 특징 %d" % (len(truth), len(FEAT)), flush=True)


def train_rows(Lsets, rng):
    """L 안의 쌍(a,b 모두 L), 모든 시각 묶음, 음성 15% 표본."""
    Xs, ys = [], []
    for f in ("F13", "F47"):
        r = REC[f]; Li = np.array(sorted(r["idx"][d] for d in Lsets[f]))
        for c in CUTS:
            A, B = np.meshgrid(Li, Li, indexing="ij"); A, B = A.ravel(), B.ravel(); keep = A != B; A, B = A[keep], B[keep]
            y = np.array([truth.get((f, r["D"][b])) == r["D"][a] for a, b in zip(A, B)])
            sel = y | (rng.random(len(y)) < .15)
            Xs.append(np.column_stack([r["F"][c][k][A[sel], B[sel]] for k in FEAT])); ys.append(y[sel])
    return np.vstack(Xs), np.concatenate(ys).astype(int)


def model_series(validator):
    m = SG[SG.validator == validator].copy()
    m["m"] = m[["sg_%d" % s for s in SGS]].mean(axis=1)
    return pd.concat([m.set_index("row_id").m, S14])


results = {}
for V in ("DIAG10", "DIAG10y"):
    fold = FOLD[V]; MS = model_series(V)
    tgt_all = SG[SG.validator == V].groupby(["farm", "day"]).validation_fold.first()
    vals = {}  # (ens, farm, day, hour) -> v (None 가능)
    for k in sorted(tgt_all.unique()):
        hid = {key for key, kk in fold.items() if kk == k}
        purge = {(f, d + j) for f, d in hid for j in (-1, 1)}
        Lsets = {f: {d for d in REC[f]["D"] if not REC[f]["isT"][REC[f]["idx"][d]] and (f, d) not in hid and (f, d) not in purge and (f, d) not in LOCK}
                 for f in ("F13", "F47")}
        Usets = {f: {d for d in REC[f]["D"] if REC[f]["isT"][REC[f]["idx"][d]] or ((f, d) in hid and d >= 179)} for f in ("F13", "F47")}
        Xt, yt = train_rows(Lsets, np.random.default_rng(k))
        models = {e: [lgb.LGBMClassifier(n_estimators=300, learning_rate=.04, num_leaves=15, min_child_samples=20, subsample=.8,
                                         subsample_freq=1, colsample_bytree=.8, scale_pos_weight=5, verbose=-1, random_state=s).fit(Xt, yt)
                      for s in ss] for e, ss in ENS.items()}
        for f in ("F13", "F47"):
            r = REC[f]; Ls, Us = Lsets[f], Usets[f]
            def prob(e, b, c):
                bi = r["idx"][b]; cand = [d for d in sorted(Ls | {u for u in Us if u < b}) if d != b]
                ai = np.array([r["idx"][d] for d in cand])
                Xc = np.column_stack([r["F"][c][kf][ai, bi] for kf in FEAT])
                p = np.mean([m.predict_proba(Xc)[:, 1] for m in models[e]], axis=0)
                j = int(np.argmax(p)); return cand[j], p[j]
            for e in ENS:
                lev = {}
                def v23(q):
                    qi = r["idx"][q]
                    if q in Ls:
                        return r["piv"]["sub_ec"][qi, 23]
                    if lev.get(q) is None:
                        return None
                    mq = np.array([MS.get("%s_%03d_%02d" % (f, q, h), np.nan) for h in range(24)])
                    return lev[q] + (mq[23] - np.nanmean(mq))
                for d in sorted(Us):
                    md = np.array([MS.get("%s_%03d_%02d" % (f, d, h), np.nan) for h in range(24)])
                    is_t = (f, d) in tgt_all.index
                    dec = {}
                    for c in (CUTS if is_t else [23]):  # 대상은 묶음마다, 그 밖(실제 평가 등)은 23시 판단만(전달용)
                        q, pq = prob(e, d, c)
                        dec[c] = v23(q) if pq >= .5 else None
                    if is_t:
                        for h in range(24):
                            vals[(e, f, d, h)] = dec[max(cc for cc in CUTS if cc <= h)]
                    pm = np.nanmean(md)
                    lev[d] = dec[23] if (dec[23] is not None and abs(dec[23] - pm) <= .30) else pm
        print("  %s 폴드 %d 완료" % (V, k), flush=True)
    results[V] = vals


def rm(x):
    return np.sqrt(np.mean(np.square(x)))
verdict = []
for V in ("DIAG10", "DIAG10y"):
    vals = results[V]; S0 = SG[SG.validator == V].copy()
    print("\n=== %s 2차 46일 ===" % V)
    cells, preds = [], []
    for e in ENS:
        for s in SGS:
            S = S0[["row_id", "farm", "day", "hour", "sub_ec", "sg_%d" % s]].rename(columns={"sg_%d" % s: "sg"}).sort_values(["farm", "day", "hour"]).copy()
            S["pm"] = S.groupby(["farm", "day"]).sg.transform(lambda x: x.expanding().mean())
            S["v"] = [vals.get((e, f, d, h)) for f, d, h in zip(S.farm, S.day, S.hour)]
            S["v"] = S.v.astype(float)
            use = S.v.notna() & ((S.v - S.pm).abs() <= .30)
            S["pred"] = np.where(use, S.sg + (S.v - S.pm), S.sg)
            a, b = rm(S.sg - S.sub_ec), rm(S.pred - S.sub_ec)
            print("  묶음 %s × SG2 %4d | SG2 %.4f → %.4f (%+.1f%%) 사용 행 %d" % (e, s, a, b, 100 * (b / a - 1), use.sum()))
            cells.append(b < a); preds.append(S)
    A = pd.concat(preds).groupby("row_id").agg(farm=("farm", "first"), day=("day", "first"), y=("sub_ec", "first"), sg=("sg", "mean"), pred=("pred", "mean")).reset_index()
    A["ydm"] = A.groupby(["farm", "day"]).y.transform("mean")
    for nm, mm in (("시드평균", A.day > 0), ("  일반", A.ydm < 1), ("  고EC", A.ydm >= 1), ("  F13", A.farm == "F13"), ("  F47", A.farm == "F47")):
        q = A[mm]; print("  %-8s SG2 %.4f → %.4f (%+.1f%%)" % (nm, rm(q.sg - q.y), rm(q.pred - q.y), 100 * (rm(q.pred - q.y) / rm(q.sg - q.y) - 1)))
    A["blk"] = A.farm + "_" + (A.day // 5).astype(str)
    B = A.groupby("blk")[["sg", "pred", "y"]].apply(lambda g: pd.Series({"a": ((g.sg - g.y) ** 2).sum(), "b": ((g.pred - g.y) ** 2).sum()}))
    idx = np.random.default_rng(1).integers(0, len(B), (2000, len(B)))
    pw = np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1))
    fok = all(rm(A[A.farm == f].pred - A[A.farm == f].y) < rm(A[A.farm == f].sg - A[A.farm == f].y) for f in ("F13", "F47"))
    ok = all(cells) and fok and pw < .025
    print("  9칸 개선 %d/9, 두 온실 %s, P(worse) %.4f → %s" % (sum(cells), "개선" if fok else "미달", pw, "충족" if ok else "미충족"))
    verdict.append(ok)
print("\n[LINK5 고정 기준] %s" % ("통과" if all(verdict) else "불합격"))
pd.DataFrame([dict(V=V, ens=k[0], farm=k[1], day=k[2], hour=k[3], v=v) for V in results for k, v in results[V].items()]).to_csv(
    os.path.join(R, "..", "results", "ec_link05_values_v1.csv"), index=False, encoding="utf-8-sig")
