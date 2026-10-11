from load import *
D=load()
D['key']=D.farm+'_'+D.rec.astype(str)
# arrays per record
recs={k:g.set_index('h').reindex(range(24)) for k,g in D.groupby('key')}
EV={'보온커튼 열림':('act_thermal','up',50),'보온커튼 닫힘':('act_thermal','down',50),
    '차광커튼 열림':('act_shade','up',50),'차광커튼 닫힘':('act_shade','down',50),
    '난방 켜짐':('act_heating','on',0),'난방 꺼짐':('act_heating','off',0),
    'CO2공급 시작':('act_co2','on',0),'천창 열림':('act_vent','on',0),'천창 닫힘':('act_vent','off',0),
    '포그 시작':('act_fog','on',0),'유동팬 켜짐':('act_circfan','on',0),'유동팬 꺼짐':('act_circfan','off',0)}
def flags(g,col,kind,th):
    a=g[col].values; f=np.zeros(24,bool)
    for t in range(1,24):
        p,c=a[t-1],a[t]
        if np.isnan(p) or np.isnan(c): continue
        if kind=='up': f[t]= p<th and c>=th
        if kind=='down': f[t]= p>=th and c<th
        if kind=='on': f[t]= p<=th and c>th
        if kind=='off': f[t]= p>th and c<=th
    return f
R=['in_temp','in_hum','in_co2','sub_temp','sub_ec','ec25']
K=[-3,0,1,2,4]
out=[];times=[]
for name,(col,kind,th) in EV.items():
    F={k:flags(g,col,kind,th) for k,g in recs.items()}
    # time distribution
    for k,f in F.items():
        g=recs[k]
        for t in np.where(f)[0]: times.append((name,g.src.dropna().iloc[0],t))
    rows=[]
    for k,g in recs.items():
        if g.src.dropna().iloc[0]!='train': continue
        f=F[k]
        for t0 in range(4,20):
            is_ev=f[t0]; near=f[max(0,t0-3):t0+5].any()
            if not is_ev and near: continue
            base=g.iloc[t0-1]
            d={'ev':is_ev,'h':t0,'farm':k[:3]}
            for r in R:
                for kk in K:
                    d[f'{r}{kk:+d}']=g[r].iloc[t0+kk]-base[r]
            rows.append(d)
    E=pd.DataFrame(rows)
    ev=E[E.ev]; 
    if len(ev)<5: out.append({'event':name,'n':len(ev)}); continue
    ctl=E[~E.ev]
    # matched by farm,h : control mean per (farm,h), subtract
    cols=[c for c in E.columns if c[:-2] in R or c[:-3] in R]
    cm=ctl.groupby(['farm','h'])[cols].mean()
    m=ev.join(cm,on=['farm','h'],rsuffix='_c')
    res={'event':name,'n':len(ev)}
    for c in cols:
        dd=(m[c]-m[c+'_c']).dropna()
        res[c]=dd.mean(); res[c+'_se']=dd.std()/np.sqrt(len(dd))
    out.append(res)
O=pd.DataFrame(out).set_index('event')
pd.set_option('display.width',300)
for r in R:
    sub=O[['n']+[f'{r}{k:+d}' for k in K]].copy()
    for k in K: sub[f'{r}{k:+d}']=sub[f'{r}{k:+d}'].round(3).astype(str)+'±'+O[f'{r}{k:+d}_se'].round(3).astype(str)
    print('==',r,'(사건−대조, 기준=사건 직전 시각)');print(sub.to_string())
T=pd.DataFrame(times,columns=['ev','src','h'])
q=T.groupby(['ev','src']).h.describe()[['count','25%','50%','75%']]
# per-record counts
print(q.to_string())
