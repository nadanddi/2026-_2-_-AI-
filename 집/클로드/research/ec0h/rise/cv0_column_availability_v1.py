# CV0: F13/F47 학습·평가 입력 열별 결측률·고유값 수 — 아직 안 쓴 열 찾기
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd
for nm in ("train_X.csv", "test_X.csv"):
    X = pd.read_csv(os.path.join(env.DATA, nm)); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
    for f in ("F13", "F47"):
        Z = X[X.row_id.str[:3] == f]
        print(nm, f, len(Z), " ".join(f"{c}:{Z[c].notna().mean():.2f}/{Z[c].nunique()}" for c in Z.columns[1:]))
