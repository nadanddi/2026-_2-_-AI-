# -*- coding: utf-8 -*-
"""MT1 (2026-10-12 집 클로드; 사용자: '정확도·포괄률 더 잘 잇는 법').  서술(신호 확인).
학습된 투표(메타 모델): 기록마다 후보 = 4모델(LM1: v7, DF, C 자정연속, D LGB)이 고른 '정답 기록' 전날들.  후보 쌍 특징:
  표 수, 각 모델이 그 후보를 골랐나·그 확률, 확률합, 0시 지문 거리 fp, 그 기록의 후보 중 fp 순위·1·2등 차, |dT_air|, 대상 2차 여부, 다른 ID 여부, 후보 수.
  라벨 = 후보 == 진단용 진짜 전날.  모델 = 로지스틱(표준화) 과 LightGBM(작게), 거울 CV 덩어리(10번호 단위·농장) 그룹 5겹 교차예측.
기록마다 메타 확률 최고 후보 하나 → 문턱별 정확도·포괄률(이을 수 있는 171 기준) 곡선, AG9 TV(정확 .836, 포괄 .92) 와 비교.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
import env_extra  # noqa
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
idx = {rid(i): i for i in range(N)}
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
LM = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "lm1_choices.csv"))
vis = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0]; m = s + sh
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
        for b in blk:
            if TRAIN[b]: vis[rid(b)] = (TRAIN & ~touch, f"{f}_{s}")
can = {r for r, (v, _) in vis.items() if r in TP and v[idx[TP[r]]]}
rows = []
for r, G in LM.groupby("rec"):
    if r not in vis: continue
    v, grp = vis[r]; b = idx[r]
    cands = [p for p in G.prev.unique() if v[idx[p]]]
    if not cands: continue
    fps = {p: fpd(b, idx[p]) for p in cands}; order = sorted(fps.values())
    for p in cands:
        sel = G[G.prev == p]; mm = set(sel.model)
        rows.append(dict(rec=r, grp=grp, cand=p, votes=len(mm), psum=sel.p.sum(), **{f"m_{k}": float(k in mm) for k in ("A_v7", "B_DF", "C_NNMID", "D_LGB")},
                         **{f"p_{k}": float(sel[sel.model == k].p.sum()) for k in ("A_v7", "C_NNMID", "D_LGB")},
                         fp=fps[p], fp_rank=sorted(fps.values()).index(fps[p]), fp_gap=(order[1] - order[0]) if len(order) > 1 else 1.0,
                         dT=abs(IN_[b, 0, 0] - IN_[idx[p], 23, 0]), p2=float(NUM[b] >= 179), xf=float(p[:3] != r[:3]), ncand=len(cands), y=float(TP.get(r) == p)))
Dm = pd.DataFrame(rows).fillna({"dT": 9.0, "fp": 4.0}).fillna(0.0); F = [c for c in Dm.columns if c not in ("rec", "grp", "cand", "y")]
grps = Dm.grp.unique(); rng = np.random.default_rng(1); fold = {g: k for k, g in enumerate(rng.permutation(grps))}; Dm["fold"] = Dm.grp.map(fold) % 5
for name in ("LR", "LGB"):
    Dm[name] = np.nan
    for k in range(5):
        tr, te = Dm[Dm.fold != k], Dm[Dm.fold == k]
        if name == "LR":
            sc = StandardScaler().fit(tr[F]); mdl = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[F]), tr.y); pr = mdl.predict_proba(sc.transform(te[F]))[:, 1]
        else:
            mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=7, min_child_samples=15, verbose=-1, random_state=3).fit(tr[F], tr.y); pr = mdl.predict_proba(te[F])[:, 1]
        Dm.loc[te.index, name] = pr
print(f"candidate pairs {len(Dm)}, records {Dm.rec.nunique()}, linkable {len(can)}")
for name in ("LR", "LGB"):
    B = Dm.loc[Dm.groupby("rec")[name].idxmax()]
    print(f"\n{name}: threshold -> anchors, precision, recall(of {len(can)})")
    for t in (0, .05, .1, .15, .2, .3, .5, .7, .9):
        S = B[B[name] >= t]; ok = int(S.y.sum())
        print(f"   >= {t:.1f}: {len(S):3d} anchors, precision {ok/len(S) if len(S) else float('nan'):.3f}, recall {ok/len(can):.2f}")
print("\nreference AG9 TV: 189 anchors, precision .836, recall .92 | v7: 54, .852, .27")
Dm.to_csv(os.path.join(env.LOCAL, "linker_v9", "mt1_meta.csv"), index=False)
