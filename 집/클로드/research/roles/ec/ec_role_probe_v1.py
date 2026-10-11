# -*- coding: utf-8 -*-
"""EC 역할 탐색 v1: 하루 안 EC 모양(관수 흔적)과 수준/변화의 관계, 입력으로 예측 가능한지.
학습 정답만 사용(분석용). 1차 = 미리 정한 3개 검사, 그 외는 2차 표시."""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import os, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
y = pd.read_csv(os.path.join(env.DATA, 'train_y.csv'))
d = X.merge(y, on='row_id')
d['farm'] = d.row_id.str[:3]; d['day'] = d.row_id.str[4:7].astype(int); d['hour'] = d.row_id.str[8:10].astype(int)
E = d.pivot_table(index=['farm', 'day'], columns='hour', values='sub_ec')
E = E.dropna()
def daily(col, fn):
    return d.groupby(['farm', 'day'])[col].agg(fn)
# VPD 근사
es = 0.6108 * np.exp(17.27 * d.in_temp / (d.in_temp + 237.3)); d['vpd'] = es * (1 - d.in_hum / 100)
F = pd.DataFrame({
    'rad_out': daily('out_rad', 'sum'), 'rad_in': daily('in_rad', 'sum'), 'tin': daily('in_temp', 'mean'),
    'tout': daily('out_temp', 'mean'), 'hum': daily('in_hum', 'mean'), 'vpd': daily('vpd', 'mean'),
    'vent': daily('act_vent', 'mean'), 'co2': daily('in_co2', 'mean')})
S = pd.DataFrame(index=E.index)
S['e0'] = E[0]; S['e23'] = E[23]; S['dlt'] = E[23] - E[0]; S['rel'] = S.dlt / E[0]
S['amp'] = E.max(axis=1) - E.min(axis=1)
S['day_minus_night'] = E[list(range(9, 17))].mean(1) - E[list(range(0, 6))].mean(1)
S['jumps'] = (E.diff(axis=1).abs() > 0.05).sum(1)
S['hmin'] = E.values.argmin(1); S['hmax'] = E.values.argmax(1)
S = S.join(F)
print('기록 수', len(S))
print('\n[기술] 수준별 하루 모양 (중앙값)')
S['lv'] = pd.cut(S.e0, [-1, 0.5, 1.0, 1.5, 9], labels=['<.5', '.5-1', '1-1.5', '>=1.5'])
print(S.groupby('lv', observed=True)[['dlt', 'rel', 'amp', 'day_minus_night', 'jumps', 'hmin', 'hmax']].median().round(3).assign(n=S.lv.value_counts()))
# 평균 시간 모양(0시 대비)
prof = E.sub(E[0], axis=0).groupby(S.lv, observed=True).median().round(3)
print('\n[기술] 0시 대비 시간별 EC 변화 중앙값\n', prof.iloc[:, ::3])

# 1차 T1: 하루 변화(23시-0시)를 그날 입력으로 예측 가능한가 (Ridge 5겹, R2)
print('\n[1차 T1] 하루 변화량(dlt) ~ 그날 입력 (+e0 미사용 / 사용)')
feat = ['rad_out', 'tin', 'tout', 'hum', 'vpd', 'vent', 'co2']
Z = S.dropna(subset=feat + ['dlt'])
for tgt in ['dlt', 'rel', 'amp', 'day_minus_night']:
    for fs in [feat, feat + ['e0']]:
        r2s = []
        for seed in [0, 1, 2]:
            pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), (Z[fs] - Z[fs].mean()) / Z[fs].std(), Z[tgt], cv=KFold(5, shuffle=True, random_state=seed))
            r2s.append(1 - ((Z[tgt] - pr) ** 2).sum() / ((Z[tgt] - Z[tgt].mean()) ** 2).sum())
        print(f'  {tgt:16s} {"+e0" if "e0" in fs else "입력만"}: CV R2 = {np.mean(r2s):+.3f} (시드 {np.round(r2s,3)})')
print('  스피어만 dlt vs 입력:', {c: round(spearmanr(Z[c], Z.dlt).statistic, 3) for c in feat})
print('  스피어만 amp vs 입력:', {c: round(spearmanr(Z[c], Z.amp).statistic, 3) for c in feat})

# 같은 동 사슬
t = pickle.load(open(r'C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl', 'rb'))
recs = pickle.load(open(r'C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl', 'rb'))['recs']
key = lambda i: (recs.farm.iloc[i], int(recs.day.iloc[i]))
t = t[t.ce1 <= 0.02].copy()
t['kb'] = t.b.map(key); t['ka'] = t.a1.map(key)
t = t[t.kb.isin(S.index) & t.ka.isin(S.index)]
print('\n같은 동 연결쌍(ce1<=.02):', len(t))
P = pd.DataFrame({'a_dlt': S.loc[list(t.ka), 'dlt'].values, 'a_amp': S.loc[list(t.ka), 'amp'].values,
                  'a_e0': S.loc[list(t.ka), 'e0'].values, 'a_rad': S.loc[list(t.ka), 'rad_out'].values,
                  'a_vpd': S.loc[list(t.ka), 'vpd'].values,
                  'b_e0': S.loc[list(t.kb), 'e0'].values, 'b_dlt': S.loc[list(t.kb), 'dlt'].values}, index=pd.MultiIndex.from_tuples(list(t.kb), names=['farm','day']))
# 1차 T2: 전날 하루 모양(출렁임 amp)이 다음날 하루 변화(b_dlt)와 관련? (수준 e0 통제 위해 부분상관)
def pcor(a, b, c):
    ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return spearmanr(ra, rb).statistic
print('[1차 T2] 전날 출렁임 a_amp vs 다음날 변화 b_dlt: rho=%.3f, 부분(e0통제)=%.3f' % (spearmanr(P.a_amp, P.b_dlt).statistic, pcor(P.a_amp.values, P.b_dlt.values, P.a_e0.values)))
print('          전날 변화 a_dlt vs 다음날 변화 b_dlt: rho=%.3f' % spearmanr(P.a_dlt, P.b_dlt).statistic)
lo = P.a_e0 < 1.0
print('          (e0<1만 n=%d) a_dlt~b_dlt rho=%.3f, a_amp~b_dlt rho=%.3f' % (lo.sum(), spearmanr(P.a_dlt[lo], P.b_dlt[lo]).statistic, spearmanr(P.a_amp[lo], P.b_dlt[lo]).statistic))
print('          피어슨 a_dlt~b_dlt=%.3f' % np.corrcoef(P.a_dlt, P.b_dlt)[0,1])

# 1차 T3: 모델 0시 잔차가 전날(같은 동) 입력 일사·VPD와 관련? (잔차=예측-정답, DIAG10 CUR 시드평균)
p = pd.read_csv(r'C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv')
q = p[(p.tag == 'CUR') & (p.validator == 'DIAG10') & (p.hour == 0)].groupby(['farm', 'day'])[['sg', 'sub_ec']].mean()
q['res'] = q.sg - q.sub_ec
P = P.join(q[['res']], how='inner')
print('[1차 T3] 0시 잔차 vs 전날 같은 동 입력/모양 (n=%d)' % len(P))
for c in ['a_rad', 'a_vpd', 'a_dlt', 'a_amp']:
    print(f'   {c}: rho={spearmanr(P[c], P.res).statistic:+.3f}')
# 2차: 잔차 vs 그날 자기 입력
Q = q.join(S[['rad_out', 'vpd', 'amp', 'dlt', 'e0']].rename_axis(['farm','day']), how='inner') if False else q.join(S[['rad_out', 'vpd', 'amp', 'dlt', 'e0']], how='inner')
S2 = S.copy(); S2.index.names=['farm','day']; Q = q.join(S2[['rad_out','vpd','amp','dlt','e0','tout','vent']], how='inner')
# 입력으로 예측한 출렁임(amp_hat, 입력만)이 잔차와 관련? (하루 일찍/당일 수준 단서인지)
Zq = Q.dropna(subset=['tout','vent','vpd','rad_out'])
print('[2차] 0시 잔차 vs 입력 tout/vent(출렁임 예측 입력):', {c: round(spearmanr(Zq[c], Zq.res).statistic,3) for c in ['tout','vent']})
print('[2차] 0시 잔차 vs 그날 자기 하루 지표:', {c: round(spearmanr(Q[c], Q.res).statistic, 3) for c in ['rad_out', 'vpd', 'amp', 'dlt']})
print('[2차] 잔차 크기: |res| 중앙 %.3f; 고EC(e0>=1.5) 잔차 중앙 %.3f, 보통 %.3f' % (Q.res.abs().median(), Q.res[Q.e0 >= 1.5].median(), Q.res[Q.e0 < 1.5].median()))
