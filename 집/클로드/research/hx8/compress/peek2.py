import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *
pa = dict(zip(T.b.astype(int), T.a1.astype(int)))
print(T[["c1","cert"]].describe())
for k in ["F13_217","F13_127","F13_125","F13_123","F13_121","F13_120","F13_118","F47_229","F47_231","F47_132","F47_134","F13_241","F13_243"]:
    i = rid2i[k]; a = pa.get(i)
    row = T[T.b == i]
    print(k, "a1", r.rid[a] if a is not None else None, "c1 %.2f"%row.c1.values[0] if len(row) else "", "cert", row.cert.values[0] if len(row) else "", "seg", D[D.rec==k].seg.values, "day", D[D.rec==k].day.values, "y %.2f"%ydm[i])
print(D.groupby("seg").day.agg(["min","max","count"]))
