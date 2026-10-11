# HR1 (사용자 질문: 0시 대신 1·2시 예측?): DIAG10 CUR OOF 의 시각별 RMSE, 0시 오차와 h시 오차의 상관, 하루 안 시간당 EC 변화
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; Q = O.groupby(["rec", "hour"]).sg.mean().unstack(); Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
E = Y - Q
for part, s in (("pass-1", Q.index.str[4:7].astype(int) < 179), ("pass-2", Q.index.str[4:7].astype(int) >= 179)):
    e = E[s]; rm = np.sqrt((e ** 2).mean())
    print(part, "n", int(s.sum()), "| RMSE by hour:", " ".join(f"{h}:{rm[h]:.3f}" for h in (0, 1, 2, 3, 6, 12, 18, 23)), "| all %.3f" % np.sqrt((e.values ** 2).mean()))
    print("   corr(err0, err_h):", " ".join(f"{h}:{e[0].corr(e[h]):.2f}" for h in (1, 2, 3, 6, 12, 23)))
    print("   share of day SSE that is the day-mean offset: %.2f" % ((e.mean(1) ** 2).sum() * 24 / (e ** 2).sum().sum()))
dY = Y.diff(axis=1).abs(); print("median |EC(h)-EC(h-1)| h=1,2,3:", [round(float(dY[h].median()), 3) for h in (1, 2, 3)])
