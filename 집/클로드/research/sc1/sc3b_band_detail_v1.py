# SC3b (사후 서술): QONLY 기록 이득을 보정 폭 10~20 / 20 초과로 더 쪼갬 (평가 기록 보정 폭은 대부분 15 이하라 그 구간이 중요)
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
P = pd.read_csv(r"sc1\sc3_records.csv"); P = P[P.arm == "QONLY"]
P["band"] = pd.cut(P.ds.abs(), [-.1, 5, 10, 15, 20, 100], labels=["<=5", "5-10", "10-15", "15-20", ">20"])
T = P.groupby(["validator", "band", "rec"]).dsse.mean().reset_index()
print(T.groupby(["validator", "band"]).agg(n=("dsse", "size"), mean=("dsse", "mean"), improved=("dsse", lambda x: float((x < -1e-9).mean()))).round(3).to_string())
print(T[(T.band.isin(["10-15", "15-20"]))].round(3).to_string(index=False))
