import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd
F=pd.read_csv(r"C:\work\farmai\집\클로드\research\ev_validator\ev_folds_v1.csv")
print(F.role.value_counts()); print(F.groupby(['fold','role']).size().unstack())
print(F[F.fold==0].sort_values(['farm','num']).to_string(max_rows=80))
