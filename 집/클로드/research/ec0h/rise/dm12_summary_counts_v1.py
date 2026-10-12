import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd
T = pd.read_csv("dm11_calendar_v1.csv")
E = T[T.split == "eval"]
print("eval by kind:", E.kind.value_counts().to_dict())
pw = T[T.kind == "순수 2차 창"]; print("pure window records:", len(pw), "| eval", int((pw.split == "eval").sum()), "| train", sorted(pw[pw.split == "train"].rec))
for r in ("F47_178", "F47_074"): print(r, T[T.rec == r][["day", "kind"]].values.tolist())
for f in ("F13", "F47"):
    P = T[(T.farm == f) & (T.pass_ == 1)]; print(f, "pass-1 numbers", P.num.min(), "-", P.num.max(), "count", len(P))
