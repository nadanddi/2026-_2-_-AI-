# -*- coding: utf-8 -*-
"""LM1 (2026-10-12 집 클로드; 사용자: '날짜랑 동만 이어주는 괜찮은 모델 몇 개 골라/새로 만들어 — 투표용').  서술(모델 만들기 + 거울 CV 정확도·다양성).
모두 0시 판단, 후보 = v7 후보 집합(features9: 같은 ID 앞 번호 + 다른 ID 학습 기록, 같은 날짜 제외), 거울 CV 기하 = cv400_v7m_critic 과 같음.
  C  NN-MID (학습 없음): 점수 = oj(외기 자정 연속 비용) + Σ_j min(dv_j, 6)(실내·구동기 23→0시 차/표준편차) 최소.  확률 = softmax(−점수).
  D  LGB-PAIR: LightGBM 이진 분류(진짜 전날 = 1), 특징 FEATS9(+fp), 덩어리 밖 학습 그룹으로 매 덩어리 재학습(같은 touch 규칙), 그룹 안 정규화 확률.
  (A v7m, B DF 는 저장 파일.)
보고: 모델별 1순위 적중률(진실 알려진 기록), '정답 기록을 고르고 확률 ≥ .8' 수·정확도, 2차 적중률, 모델 쌍 일치율·일치 시 정확도.
출력: local/linker_v9/lm1_choices.csv (rec, model, prev, p).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
from feat9 import *
import env_extra  # noqa
import lightgbm as lgb
OUT = os.path.join(env.LOCAL, "linker_v9")
TRG = train_groups9(0)   # (b, a, Xc, y) with FEATS9
FI = {f: i for i, f in enumerate(FEATS9)}
rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        tr = [(X_, y_) for b_, a_, X_, y_ in TRG if not touch[b_] and not touch[a_]]
        Xt = np.vstack([x for x, _ in tr]); yt = np.concatenate([(np.arange(len(x)) == y).astype(int) for x, y in tr])
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=20, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=7)
        mdl.fit(Xt, yt)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b]: continue
            df, _ = features9(b, 0, lab, removed=rem2)
            if len(df) == 0: continue
            Xc = df[FEATS9].values.astype(float); a = df.a.values.astype(int)
            sc = Xc[:, FI["oj"]] + np.minimum(Xc[:, [FI[f"v{j}"] for j in range(8)]], 6).sum(1)
            pc = np.exp(-(sc - sc.min())); pc /= pc.sum(); k = int(np.argmax(pc))
            rows.append(dict(rec=rid(b), model="C_NNMID", prev=rid(a[k]), p=float(pc[k]), lab=bool(lab[a[k]])))
            pd_ = mdl.predict_proba(Xc)[:, 1]; pd_ = pd_ / pd_.sum(); k = int(np.argmax(pd_))
            rows.append(dict(rec=rid(b), model="D_LGB", prev=rid(a[k]), p=float(pd_[k]), lab=bool(lab[a[k]])))
L = pd.DataFrame(rows)
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
L = pd.concat([L, V7.assign(model="A_v7", lab=V7.prev_is_labelled.astype(bool))[["rec", "model", "prev", "p", "lab"]]])
DF = pd.read_csv(os.path.join(OUT, "df1_picks_mirror.csv"))
L = pd.concat([L, DF.assign(model="B_DF", p=1.0, lab=True)[["rec", "model", "prev", "p", "lab"]]])
L.to_csv(os.path.join(OUT, "lm1_choices.csv"), index=False)
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
L["truth"] = L.rec.map(TP); L = L[L.truth.notna()]; L["ok"] = L.prev == L.truth; L["p2"] = L.rec.str[4:7].astype(int) >= 179
pd.set_option("display.width", 200)
print("per model (records with known truth):")
for mname, G in L.groupby("model"):
    c = G[G.lab & (G.p >= .8)]
    print(f"  {mname:8s} n {len(G):3d} | top-1 hit {G.ok.mean():.3f} | pass-2 hit {G[G.p2].ok.mean():.3f} | labelled & p>=.8: {len(c)} acc {c.ok.mean() if len(c) else float('nan'):.3f}")
W = L.pivot_table(index="rec", columns="model", values="prev", aggfunc="first"); OK = L.pivot_table(index="rec", columns="model", values="ok", aggfunc="first")
ms = list(W.columns)
print("\npairwise: agreement rate | accuracy when agree (both present)")
for i in range(len(ms)):
    for j in range(i + 1, len(ms)):
        a, b = ms[i], ms[j]; m = W[a].notna() & W[b].notna(); ag = W[a][m] == W[b][m]
        print(f"  {a:8s} vs {b:8s}: n {int(m.sum())}, agree {ag.mean():.2f}, acc when agree {OK[a][m][ag].astype(bool).mean():.3f}, acc of {a} when disagree {OK[a][m][~ag].astype(bool).mean():.3f}")
