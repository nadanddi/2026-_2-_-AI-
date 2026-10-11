# 확인용: 12회차 제출본의 해당 기록 시각별 EC 예측 (읽기만)
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
for r in ("F47_233", "F47_221", "F13_228", "F13_221"):
    v = S[S.rec == r].sort_values("h").sub_ec.values; print(r, " ".join(f"{x:.2f}" for x in v))
