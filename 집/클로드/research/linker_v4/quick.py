from feat4 import *
import time; t=time.time()
b=int(np.where((FARM=="F47")&(NUM==191))[0][0]); lab=TRAIN.copy(); lab[(FARM=="F47")&(np.abs(NUM-191)<=1)]=False
df,info=features(b,0,lab); print(time.time()-t, info, len(df))
print(df.sort_values("oj").head(5)[["a","oj","isprev","dsdev","dsna","excl"]].assign(rec=lambda d:[rid(i) for i in d.a]))
print("F47_057 row:",df[df.a==int(np.where((FARM=="F47")&(NUM==57))[0][0])][["isprev","dsdev","dsna","oj"]])
