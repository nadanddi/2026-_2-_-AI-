from common import *
A,S=load()
for f in s2_folds(A): print(f[0], f[1][0], f[1][-1], len(f[1]))
for g in ["F13","F47"]:
    T=day_table(A,g)
    t=T[T.test==1]; print(g,"test days with earlier twin:", t.twin.notna().sum(), "/", len(t), " train S2 days w/ twin:", T[(T.test==0)&(T.day>=177)].twin.notna().sum(), "/", ((T.test==0)&(T.day>=177)).sum())
