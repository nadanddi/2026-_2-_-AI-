# SC2c (서술): 평가 60기록 / 2차 학습 기록의 현재 계절 값(season_interp/used) vs 같은 날짜 1차 기록 번호(같은 농장, DM11 달력) 차이
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
C = pd.read_csv(r"ec0h\rise\dm11_calendar_v1.csv")
p1 = C[(C.pass_ == 1) & C.day.notna()]
def same_date_p1(f, n):
    d = C[(C.farm == f) & (C.num == n)].day.values
    if not len(d) or np.isnan(d[0]): return np.nan
    s = p1[(p1.day == d[0]) & (p1.farm == f)].num
    return float(s.median()) if len(s) else np.nan
E = pd.read_csv(r"audit_ec\a1_eval_season.csv"); E["exact"] = [same_date_p1(f, d) for f, d in zip(E.farm, E.day)]
E["d"] = E.exact - E.season_interp
print("eval records with same-farm same-date pass-1 record:", int(E.exact.notna().sum()), "/", len(E))
print(E[E.exact.notna()][["farm", "day", "season_interp", "exact", "d"]].round(1).sort_values("d", key=abs, ascending=False).to_string(index=False))
T = pd.read_csv(r"audit_ec\a1_train_p2_season.csv"); T["exact"] = [same_date_p1(f, d) for f, d in zip(T.farm, T.day)]; T["d"] = T.exact - T.season_used
print("\ntraining pass-2 largest differences:"); print(T[T.exact.notna()].sort_values("d", key=abs, ascending=False).head(10)[["farm", "day", "season_used", "exact", "d"]].round(1).to_string(index=False))
