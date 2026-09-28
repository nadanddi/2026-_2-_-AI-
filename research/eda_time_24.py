"""Q4: control rules - heating vs in_temp, thermal curtain vs hour/radiation, per source cluster."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX])
for f in ['F13','F47']:
    L=pd.read_csv(f'local/eda_time_10_{f}.csv').set_index('day')
    d=X[X.farm==f].copy(); d['cl']=d.day.map(L.cl)
    d['on']=d.act_heating>0
    d['bin']=pd.cut(d.in_temp,[-5,6,8,10,12,14,16,18,20,25,50])
    print(f,'P(heating>0) by in_temp bin x cluster'); print(d.pivot_table(index='bin',columns='cl',values='on',aggfunc='mean').round(2).to_string())
    # heating level vs in_temp correlation within night hours
    n=d[(d.hour<=5)|(d.hour>=20)]
    print(' night corr(heating, in_temp) by cluster', n.groupby('cl').apply(lambda s: round(s.act_heating.corr(s.in_temp),2)).to_dict())
    # is heating tied to circfan?
    print(' share heating==circfan', n.groupby('cl').apply(lambda s: round((s.act_heating==s.act_circfan).mean(),2)).to_dict())
    # thermal curtain by hour
    print(' act_thermal median by hour (cl rows pooled):', d.groupby('hour').act_thermal.median().astype(int).tolist())
    print(' act_thermal most common values:', d.act_thermal.round(2).value_counts().head(8).to_dict())
    # does the heater respond to previous hour's temperature dropping below a setpoint? distribution of in_temp when heating switches on
    d=d.sort_values('t'); d['on_prev']=d.on.shift(1)
    sw=d[(d.on)&(d.on_prev==False)]
    print(' in_temp at switch-on: median %.2f IQR %.2f-%.2f; night only median %.2f'%(sw.in_temp.median(),sw.in_temp.quantile(.25),sw.in_temp.quantile(.75), sw[(sw.hour<=6)|(sw.hour>=18)].in_temp.median()))
