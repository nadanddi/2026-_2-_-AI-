from feat4 import *
b=int(np.where((FARM=="F47")&(NUM==191))[0][0]); vis=visible(b,set()); comp=components(vis); gpos,mp=calendar(vis,comp)
by={}
for i in np.where(vis)[0]: by.setdefault(comp[i],[]).append(rid(i))
for c,p in sorted(gpos.items(),key=lambda x:x[1]):
    if 48<p<62: print(round(p,2),by[c])
print([ (n,round(mp(int(np.where((FARM=="F47")&(NUM==n))[0][0])),1)) for n in [50,52,54,56,57,58,60]])
