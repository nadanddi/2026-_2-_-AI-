# EV10b (EV10 사후 서술): v7 과 v10 앵커가 다른 기록별 EC·온도 오차 변화(EV, 시드 7 / 칸 7A), 2차 기록 표시.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("ev10_anchor_score_v1.py", encoding="utf-8").read().split("res = []; used7")[0])
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
G0 = R[R.seed.astype(str) == "7"].sort_values(["rec", "hour"]).reset_index(drop=True); cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
q18 = D["v18_a_7A"].values; base_t = D["v22_a_7A"].values; nt, _ = temp_pred("7A", A10)
rows = []
for rec in sorted(set(A7) | set(A10)):
    if A7.get(rec) == A10.get(rec): continue
    g = G0[G0.rec == rec]; t = D[D.rec == rec]
    if len(g) == 0: continue
    k = int(g.fold.iloc[0]); ix = g.index.values
    def ec_pred(A):
        if rec not in A or A[rec] in hid[k]: return G0.SG2.values[ix]
        return np.clip(G0.SG2.values[ix] + (EC[idx[A[rec]], 23] - cm[ix]), lo, hi)
    y = g.y.values; e7 = ((ec_pred(A7) - y) ** 2).sum(); e10 = ((ec_pred(A10) - y) ** 2).sum()
    ti = t.index.values; yt = t.y.values; t7 = ((base_t[ti] - yt) ** 2).sum(); t10 = ((nt[ti] - yt) ** 2).sum()
    rows.append(dict(rec=rec, p2=int(rec[4:7]) >= 179, v7=A7.get(rec, "-"), v10=A10.get(rec, "-"), truth=TP.get(rec, ""), dEC=e10 - e7, dTEMP=t10 - t7))
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(T.sort_values("dEC", ascending=False).round(3).to_string(index=False))
print("\nsum dEC pass-2 %.3f / pass-1 %.3f | sum dTEMP pass-2 %.2f / pass-1 %.2f" % (T[T.p2].dEC.sum(), T[~T.p2].dEC.sum(), T[T.p2].dTEMP.sum(), T[~T.p2].dTEMP.sum()))
