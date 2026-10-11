from load import *
D=load(); D['key']=D.farm+'_'+D.rec.astype(str)
A=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
G=D.groupby('key')
for c in ['in_temp','in_hum','in_co2','sub_temp','sub_ec','ec25','out_rad','out_temp']+A: D['d_'+c]=G[c].diff()
D['act_chg']=D[['d_'+a for a in A]].abs().max(axis=1)
D['act_chg2']=G['act_chg'].shift(1).fillna(0).clip(lower=0)  # previous-hour change
D['quiet']=(D.act_chg<=2)&(D.act_chg2<=2)
D['night']=(D.h>=20)|(D.h<=5)
tr=D.src=='train'
def rep(name,m,extra=None):
    n=m.sum(); ntr=(m&tr).sum(); nte=(m&~tr).sum()
    rtr=ntr/(tr&D.d_in_temp.notna()).sum()*1000; rte=nte/((~tr)&D.d_in_temp.notna()).sum()*1000
    nrec=D[m].key.nunique()
    s=f"{name}: 학습 {ntr}건({rtr:.1f}/천시간), 평가입력 {nte}건({rte:.1f}/천시간), 기록수 {nrec}"
    if extra is not None: s+=' | '+extra
    print(s)
    return m
m=rep('A 구동기 2시간 무변화인데 실내온도 ±3℃/h 이상',D.quiet&(D.d_in_temp.abs()>=3)&(D.d_out_rad.abs()<100))
print('   시각분포',D[m].h.value_counts().sort_index().to_dict(),' 방향 +',(D[m].d_in_temp>0).sum())
m=rep('A2 밤 구동기 무변화 실내온도 ±2℃/h',D.quiet&D.night&(D.d_in_temp.abs()>=2))
print('   ', D[m][['key','h','in_temp','d_in_temp','out_temp','d_out_temp','act_heating']].head(8).to_string())
m=rep('B 배지온도 ±1℃/h인데 실내온도 변화<0.5℃',(D.d_sub_temp.abs()>=1)&(D.d_in_temp.abs()<0.5))
print('   시각',D[m].h.value_counts().sort_index().to_dict(),'방향+',(D[m].d_sub_temp>0).sum(),'동시 EC변화>0.02:',(D[m].d_sub_ec.abs()>0.02).sum())
m=rep('B2 배지온도 1시간 1.5℃ 이상 하락(관수 냉수 흔적 후보)',(D.d_sub_temp<=-1.5))
print('   시각',D[m].h.value_counts().sort_index().to_dict(),' 그때 EC25변화 중앙',D[m].d_ec25.median().round(3) if m.any() else None,' 실내온도변화 중앙',D[m].d_in_temp.median())
m=rep('C 밤(20~5시) CO2 +100ppm/h 이상, CO2공급 0',D.night&(D.d_in_co2>=100)&(D.act_co2==0))
print('   ',D[m][['key','h','in_co2','d_in_co2','act_circfan','act_vent']].head(8).to_string())
m=rep('C2 CO2 1000ppm 이상 (공급 0)',(D.in_co2>=1000)&(D.act_co2==0))
m=rep('C3 CO2 300ppm 미만(외기 이하)',(D.in_co2<300))
print('   시각',D[m].h.value_counts().sort_index().to_dict())
m=rep('D 습도 ±10%p/h, 구동기 무변화, 온도변화<1℃',D.quiet&(D.d_in_hum.abs()>=10)&(D.d_in_temp.abs()<1))
print('   시각',D[m].h.value_counts().sort_index().to_dict())
m=rep('D2 습도 100% 포화',(D.in_hum>=99))
m=rep('E 원EC 1시간 ±0.1 이상 계단',(D.d_sub_ec.abs()>=0.1))
print('   시각',D[m].h.value_counts().sort_index().to_dict(),' 상승',(D[m].d_sub_ec>0).sum(),' 동시 배지온도변화 |>0.5|:',(D[m].d_sub_temp.abs()>0.5).sum(), ' 구동기 무변화:',D[m].quiet.sum())
# flatlines
def runs(s):
    r=(s!=s.shift()).cumsum(); return s.groupby(r).transform('size')
for c in ['in_temp','in_hum','in_co2','sub_temp','sub_ec','out_temp']:
    D['run_'+c]=D.groupby('key')[c].transform(runs)
    m=(D['run_'+c]>=6)&D[c].notna()
    rep(f'F {c} 같은 값 6시간 이상 연속',m)
m=rep('G 실내온도가 외기보다 낮음(밤, 1℃ 이상)',D.night&(D.in_temp<D.out_temp-1))
m=rep('G2 낮(11~15) 실내<외기',(D.h.between(11,15))&(D.in_temp<D.out_temp))
m=rep('H 결측 실내 입력(in_temp NaN)',D.in_temp.isna())
print('   기록',D[D.in_temp.isna()].groupby('key').size().to_dict())
m=rep('I 난방 100인데 실내온도 하락 ≥1℃/h',(D.act_heating>=90)&(D.d_in_temp<=-1))
print('   시각',D[m].h.value_counts().sort_index().to_dict())
m=rep('J 보온커튼 닫힘(0)인데 낮 일사 >300',(D.act_thermal<5)&(D.out_rad>300))
m=rep('K 배지온도>실내온도+5 (밤)',D.night&(D.dTs>5))
print('   ',D[m].groupby('key').size().sort_values().tail(5).to_dict())
m=rep('K2 배지온도<실내온도−10',(D.dTs<-10))
print('   시각',D[m].h.value_counts().sort_index().to_dict())
# record boundaries: discontinuity of out_temp between rec r h23 and rec r+1 h0
D2=D.sort_values(['farm','rec','h'])
b=D2[D2.h==0].copy(); prev=D2[D2.h==23].set_index(['farm','rec'])
b['prev_out']=[prev.out_temp.get((f,r-1),np.nan) for f,r in zip(b.farm,b.rec)]
b['prev_in']=[prev.in_temp.get((f,r-1),np.nan) for f,r in zip(b.farm,b.rec)]
b['prev_sub']=[prev.sub_temp.get((f,r-1),np.nan) for f,r in zip(b.farm,b.rec)]
b['prev_ec']=[prev.sub_ec.get((f,r-1),np.nan) for f,r in zip(b.farm,b.rec)]
w=D2[D2.h.between(1,23)]
print('경계(23→0시) |Δ외기온| 중앙',(b.out_temp-b.prev_out).abs().median(),' 기록 내부 1시간 |Δ외기온| 중앙',w.d_out_temp.abs().median())
print('경계 |Δ실내온| 중앙',(b.in_temp-b.prev_in).abs().median(),' 내부(0시근처 h1)',D2[D2.h==1].d_in_temp.abs().median())
print('경계 |Δ배지온| 중앙',(b.sub_temp-b.prev_sub).abs().median(),' 내부 h1',D2[D2.h==1].d_sub_temp.abs().median())
print('경계 |ΔEC| 중앙',(b.sub_ec-b.prev_ec).abs().median(),' 내부 h1',D2[D2.h==1].d_sub_ec.abs().median())
print('경계 외기 연속(|Δ|<1) 비율',((b.out_temp-b.prev_out).abs()<1).mean().round(2))
# identical out_temp vectors F13 vs F47
V={k:tuple(g.sort_values('h').out_temp.round(1)) for k,g in D.groupby('key')}
f13={v:k for k,v in V.items() if k.startswith('F13')}
same=[(k,f13[v]) for k,v in V.items() if k.startswith('F47') and v in f13]
print('F47 기록 중 외기온 24시간이 F13 어떤 기록과 완전히 같음:',len(same),'/',sum(k.startswith('F47') for k in V), same[:8])
