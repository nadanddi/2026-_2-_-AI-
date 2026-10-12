# AG5 (2026-10-12 집 클로드; 사용자: '정확도 97% 로 올리면 왜 나빠지나, 잘못 고른 날이 더 좋았나?') 서술:
# AG3 FILT 가 뺀 v7 앵커(두 방법 불일치) 각각의 가치 = 앵커 없을 때 SSE − 앵커 있을 때 SSE (EV, 온도 4칸 평균 / EC 3시드 평균), 진실상 맞음·틀림별.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
H = r"C:\work\farmai\집\클로드\research"
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv")).set_index("rec")
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
V["agree"] = V.prev == P.prev.reindex(V.index); v7a = (V.p >= .8) & V.prev_is_labelled.astype(bool) & (V.dT_air.abs() <= 1.5)
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]
cells = ["7A", "7B", "101A", "101B"]
gt = D.groupby("rec").apply(lambda g: np.mean([((g[f"v18_a_{c}"] - g.sub_temp) ** 2).sum() - ((g[f"v22_a_{c}"] - g.sub_temp) ** 2).sum() for c in cells]))
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
ge = R.groupby("rec").apply(lambda g: (((g.SG2 - g.y) ** 2).sum() - ((g.EA2 - g.y) ** 2).sum()) / g.seed.nunique())
T = V[v7a].copy(); T["truth"] = [TP.get(r, "") for r in T.index]; T["correct"] = T.prev == T.truth
T["gain_temp"] = gt.reindex(T.index); T["gain_ec"] = ge.reindex(T.index); T["applied"] = (gt.reindex(T.index).abs() > 1e-9)
T = T[T.applied]
pd.set_option("display.width", 200)
print("anchor value = SSE without anchor - SSE with anchor (positive = anchor helps), EV-applied v7 anchors:", len(T))
for (ag, ok), g in T.groupby(["agree", "correct"]):
    print(f"  DF agree={ag!s:5s} correct={ok!s:5s}: n {len(g):2d} | temp gain sum {g.gain_temp.sum():+7.2f} (mean {g.gain_temp.mean():+.2f}) | EC gain sum {g.gain_ec.sum():+.3f} (mean {g.gain_ec.mean():+.3f})")
print("\ndropped by FILT (DF disagree), per record:")
print(T[~T.agree].sort_values("gain_temp")[["prev", "truth", "correct", "p", "gain_temp", "gain_ec"]].round(3).to_string())
TL = pd.read_pickle(S3 + r"\truth.pkl"); CERT = {rid(int(b)): bool(c) for b, c in zip(TL.b, TL.cert)}
idx = {rid(i): i for i in range(N)}
print("\nlabel continuity check (diagnostic): |ST(prev,23) - ST(b,0)|, |EC(prev,23) - EC(b,0)| for v7 pick vs 'truth'")
W = T[~T.correct].copy()
for r, row in W.iterrows():
    b = idx[r]; a = idx[row.prev]; t = idx[row.truth] if row.truth else None
    s = lambda x: f"ST {abs(ST[x,23]-ST[b,0]):.2f} EC {abs(EC[x,23]-EC[b,0]):.3f}"
    print(f"  {r}: v7 {row.prev} [{s(a)}] | truth {row.truth} cert {CERT.get(r)} [{s(t) if t is not None else '-'}] | temp gain {row.gain_temp:+.2f}")
allc = T[T.correct]
print("  reference, correct anchors median: ST %.2f EC %.3f" % (np.median([abs(ST[idx[p],23]-ST[idx[r],0]) for r, p in zip(allc.index, allc.prev)]), np.median([abs(EC[idx[p],23]-EC[idx[r],0]) for r, p in zip(allc.index, allc.prev)])))
