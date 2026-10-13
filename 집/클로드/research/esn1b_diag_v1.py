import os, sys
R = r"C:\work\farmai\집\클로드\research"
sys.path.insert(0, R); os.chdir(R)
src = open("esn1_chain_esn_temp_v1.py", encoding="utf-8").read()
__file__ = os.path.join(R, "esn1_chain_esn_temp_v1.py")
exec(src[:src.index("for L, (M, cells) in FR.items():")])
M, cells = FR["DIAG10"]
key, h, y, anc, recs, pos, Q, A, Y = layout_maps(M, cells)
fold_of = {r: M.fold.values[pos[r][0]] for r in recs}
k = sorted(set(fold_of.values()))[0]
te = [r for r in recs if fold_of[r] == k]; hid = set(te); tr = [r for r in DREC if r not in purge(te)]
Xtr = build(tr, DQ, hid); Ytr = np.stack([DY[r] - DQ[r] for r in tr]); Xte = build(te, Q, hid); Yte = np.stack([Y[r] - Q[r] for r in te])
P_tr = fit_predict(Xtr, Ytr, Xtr); P_te = fit_predict(Xtr, Ytr, Xte)
m = np.isfinite(Ytr); mt = np.isfinite(Yte)
print("train: resid rmse %.3f -> after %.3f | corr %.3f" % (np.sqrt(np.nanmean(Ytr ** 2)), np.sqrt(np.nanmean((Ytr - P_tr) ** 2)), np.corrcoef(P_tr[m], Ytr[m])[0, 1]))
print("test : resid rmse %.3f -> after %.3f | corr %.3f | pred sd %.3f" % (np.sqrt(np.nanmean(Yte ** 2)), np.sqrt(np.nanmean((Yte - P_te) ** 2)), np.corrcoef(P_te[mt], Yte[mt])[0, 1], np.nanstd(P_te)))
print("n tr", len(tr), "n te", len(te), "Y NaN share tr", round(float(np.isnan(Ytr).mean()), 3), "te", round(float(np.isnan(Yte).mean()), 3))
print("Q vs DQ max diff on te:", max(np.nanmax(np.abs(Q[r] - DQ[r])) for r in te))
print("DQ vs DY sample:", np.round(DQ[tr[0]][:5], 2), np.round(DY[tr[0]][:5], 2))
