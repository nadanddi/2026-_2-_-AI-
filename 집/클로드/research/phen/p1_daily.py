from load import *
D=load()
D['grp']=D.farm+'-'+D.phase.astype(str)+np.where(D.src=='test','te','tr')
V=['out_temp','out_hum','out_rad','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','sub_temp','sub_ec','ec25','dTs']
H=[0,3,5,6,7,8,9,10,12,14,16,17,18,19,20,22]
pd.set_option('display.width',250)
for v in V:
    g=D.groupby(['grp','h'])[v]
    med=g.median().unstack()[H]; q1=g.quantile(.25).unstack()[H]; q3=g.quantile(.75).unstack()[H]
    print('==',v); 
    out=med.round(2).astype(str)+'['+q1.round(1).astype(str)+','+q3.round(1).astype(str)+']'
    print(med.round(2).to_string())
    print(' IQR width', (q3-q1).round(1).to_string().replace('\n','\n '))
# rec ids
print(D.groupby(['farm','src']).rec.agg(['min','max']))
print(D[D.phase==2].groupby(['farm','rec']).src.first().unstack(0).T.to_string())
