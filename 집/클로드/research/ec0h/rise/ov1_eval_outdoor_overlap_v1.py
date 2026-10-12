# OV1 (서술): 평가 60기록의 외기(24h×4)가 학습 기록과 얼마나 겹치나 — 완전 일치(>=90% 칸) 학습 형제 수, 없으면 가장 가까운 학습 기록 거리
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
CAL = pd.read_csv("dm11_calendar_v1.csv"); KIND = dict(zip(CAL.rec, CAL.kind)); DAY = dict(zip(CAL.rec, CAL.day))
Xf = X.reshape(N, -1); TRi = np.where(TRAIN)[0]; EVi = np.where(~TRAIN)[0]
sdz = np.nanstd(X[TRAIN].reshape(-1, 4), 0)
rows = []
for i in EVi:
    eq = (np.abs(Xf[TRi] - Xf[i]) < 1e-6).mean(1); sib = TRi[eq >= .9]
    d = np.nanmean(np.abs(X[TRi] - X[i]) / sdz, axis=(1, 2)); j = TRi[np.argmin(d)]
    e0 = (np.abs(X[TRi, 0] - X[i, 0]) < 1e-6).all(1).sum()
    rows.append(dict(rec=rid(i), day=DAY.get(rid(i)), kind=KIND.get(rid(i)), n_train_same_date=len(sib), same_date_train=" ".join(rid(s) for s in sib),
                     zero_h_exact_train=int(e0), nearest_train=rid(j), nearest_dist=float(d.min())))
D = pd.DataFrame(rows)
print("eval records with >=1 training record of the SAME date (outdoor identical):", int((D.n_train_same_date > 0).sum()), "/ 60")
print("distribution of # same-date training records:", D.n_train_same_date.value_counts().sort_index().to_dict())
print("by date kind:"); print(D.groupby("kind").n_train_same_date.agg(["size", lambda s: int((s > 0).sum()), "mean"]).to_string())
tr_pairs = []
for i in TRi:
    eq = (np.abs(Xf[TRi] - Xf[i]) < 1e-6).mean(1); tr_pairs.append(int((eq >= .9).sum()) - 1)
print("reference: training records with >=1 other training same-date record: %d / %d" % (sum(1 for x in tr_pairs if x > 0), len(TRi)))
print("\neval records WITHOUT any same-date training record (nearest training weather distance, z units; same-date = 0):")
print(D[D.n_train_same_date == 0][["rec", "day", "kind", "nearest_train", "nearest_dist"]].round(3).to_string(index=False))
dist_tr = []
for i in TRi[:400]:
    d = np.nanmean(np.abs(X[TRi] - X[i]) / sdz, axis=(1, 2)); d[TRi == i] = 9; eqm = (np.abs(Xf[TRi] - Xf[i]) < 1e-6).mean(1) >= .9; d[eqm] = 9; dist_tr.append(d.min())
print("reference: nearest DIFFERENT-date training weather distance among training records: median %.3f, 10%% %.3f" % (np.median(dist_tr), np.quantile(dist_tr, .1)))
D.to_csv("ov1_eval_overlap_v1.csv", index=False)
