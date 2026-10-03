from models_v2 import *
class ECModel:
 def fit(self,tr,core,seed):
  # Original DC4 appends season after removing day; order affects ET/MLP.
  self.full=[c for c in core.FULL if c!='day']+['season'];self.base=[c for c in core.BASE if c!='day']+['season']
  self.models=[core.et(seed),core.lg(seed,'tweedie'),make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))]
  for m,c in zip(self.models,[self.full,self.base,self.base]):m.fit(tr[c],tr.sub_ec)
  return self
 def predict(self,q):
  p=[m.predict(q[c]) for m,c in zip(self.models,[self.full,self.base,self.base])]
  return dict(raw_r3=.6*p[0]+.3*p[1]+.1*p[2],raw_ET=p[0],raw_LGB=p[1],raw_MLP=p[2])
