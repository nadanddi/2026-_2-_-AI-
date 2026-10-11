"""실내 유래 신호 3개 vs 0시 EC 잔차 (사전 고정 판정, 학습 자료만).
사전 고정: 신호별 농장 내 순위상관(잔차 vs 신호), 1차(ID 앞 절반)·2차(뒤 절반) 두 온실 모두 같은 부호,
 순열 p(농장 내 섞기, 5000회) < 0.05/3 (본페로니) 를 둘 다 만족해야 '살아남음'.
잔차 = log(sub_ec) - log(sg), tag CUR, DIAG10, 시드 평균, hour 0.
신호(인과: 0시 행 입력 + 같은 온실 이전 기록 18~23시 입력만):
 S1 co2_rate : 이전 기록 19~23시 중 CO2 공급0·환기0 시간의 in_co2 시간당 증가량 평균 (밤 호흡/밀폐도)
 S2 ah_surplus: 이전 기록 20~23시 + 0시 실내-외기 절대습도 차 평균 g/m3 (배지·식물 수분 방출 흔적)
 S3 dew_gap0 : 0시 실내 기온 - 실내 이슬점 (응결 근접도; 작을수록 응결)
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import os, json, env, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
rng = np.random.default_rng(0)
X = pd.read_csv(os.path.join(str(env.DATA), 'train_X.csv'))
p = X.row_id.str.split('_', expand=True); X['farm'] = p[0]; X['day'] = p[1].astype(int); X['hour'] = p[2].astype(int)
X = X[X.farm.isin(['F13', 'F47'])].copy()

def ah(t, rh):  # g/m3
    es = 6.112 * np.exp(17.67 * t / (t + 243.5)); return 216.7 * (rh / 100 * es) / (273.15 + t)
X['ahd'] = ah(X.in_temp, X.in_hum) - ah(X.out_temp, X.out_hum)
g = np.log(X.in_hum.clip(1, 100) / 100) + 17.67 * X.in_temp / (243.5 + X.in_temp)
X['dew_gap'] = X.in_temp - 243.5 * g / (17.67 - g)
X['dco2'] = X.groupby(['farm', 'day']).in_co2.diff()
ok = (X.act_co2.fillna(0) == 0) & (X.act_vent.fillna(0) == 0)
X['dco2_ok'] = X.dco2.where(ok & (X.groupby(['farm', 'day']).act_co2.shift(1).fillna(0) == 0))
rows = []
for (f, d), q in X.groupby(['farm', 'day']):
    q = q.set_index('hour')
    rows.append(dict(farm=f, day=d,
        co2_rate_ev=q.loc[19:23].dco2_ok.mean(),
        ahd_ev=q.loc[20:23].ahd.mean(), ahd0=q.ahd.get(0, np.nan), dew_gap0=q.dew_gap.get(0, np.nan)))
D = pd.DataFrame(rows).sort_values(['farm', 'day'])
# 이전 기록(같은 온실 직전 ID) 값 이어받기
for c in ['co2_rate_ev', 'ahd_ev']:
    D[c + '_prev'] = D.groupby('farm')[c].shift(1)
D['S1_co2_rate'] = D.co2_rate_ev_prev
D['S2_ah_surplus'] = D[['ahd_ev_prev']].assign(a0=D.ahd0).mean(axis=1, skipna=True)
D['S3_dew_gap0'] = D.dew_gap0

r = pd.read_csv(r'C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv')
r = r[(r.tag == 'CUR') & (r.validator == 'DIAG10') & (r.hour == 0)]
r = r.groupby(['farm', 'day']).agg(sg=('sg', 'mean'), y=('sub_ec', 'first')).reset_index()
r['res'] = np.log(r.y) - np.log(r.sg)
M = r.merge(D, on=['farm', 'day'])
# 0시 입력 대비 결정적 대조: 신호를 0시 in_hum/in_temp로 회귀 제거한 부분 버전도 함께
X0 = X[X.hour == 0][['farm', 'day', 'in_temp', 'in_hum', 'in_co2']]
M = M.merge(X0, on=['farm', 'day'])
M['half'] = M.groupby('farm').day.transform(lambda s: np.where(rankdata(s) <= len(s) / 2, 1, 2))
out = {'n': int(len(M)), 'n_by_farm': M.farm.value_counts().to_dict()}

def rho_within(m, c):
    m = m.dropna(subset=[c, 'res'])
    if len(m) < 8: return np.nan, len(m)
    return spearmanr(m[c], m.res)[0], len(m)

def perm_p(m, c, B=5000):
    m = m.dropna(subset=[c, 'res']).copy()
    # 농장 내 순위로 바꿔 합친 상관
    m['rx'] = m.groupby('farm')[c].rank(pct=True); m['ry'] = m.groupby('farm').res.rank(pct=True)
    obs = np.corrcoef(m.rx, m.ry)[0, 1]; cnt = 0
    xs = [m[m.farm == f].rx.values for f in ['F13', 'F47']]; ys = np.concatenate([m[m.farm == f].ry.values for f in ['F13', 'F47']])
    for _ in range(B):
        xp = np.concatenate([rng.permutation(x) for x in xs])
        cnt += abs(np.corrcoef(xp, ys)[0, 1]) >= abs(obs)
    return obs, (cnt + 1) / (B + 1)

for c in ['S1_co2_rate', 'S2_ah_surplus', 'S3_dew_gap0']:
    res = {}
    for f in ['F13', 'F47']:
        for h in [1, 2]:
            res[f'{f}_h{h}'] = rho_within(M[(M.farm == f) & (M.half == h)], c)
    obs, pp = perm_p(M, c)
    # 0시 실내 기온·습도·CO2 통제 후 (농장별 선형 잔차)
    m2 = M.dropna(subset=[c]).copy()
    for f in ['F13', 'F47']:
        s = m2.farm == f; A = np.c_[np.ones(s.sum()), m2.loc[s, ['in_temp', 'in_hum', 'in_co2']].fillna(m2.loc[s, ['in_temp', 'in_hum', 'in_co2']].mean()).values]
        m2.loc[s, c + '_part'] = m2.loc[s, c] - A @ np.linalg.lstsq(A, m2.loc[s, c].values, rcond=None)[0]
    obs2, pp2 = perm_p(m2, c + '_part')
    signs = [np.sign(v[0]) for v in res.values() if v[0] == v[0]]
    surv = (len(signs) == 4 and len(set(signs)) == 1 and pp < 0.05 / 3)
    out[c] = dict(pooled_rho=round(obs, 4), perm_p=round(pp, 4), partial_rho=round(obs2, 4), partial_p=round(pp2, 4),
                  by_farm_half={k: (round(v[0], 3), v[1]) for k, v in res.items()}, survives=bool(surv))
    # 고EC 날(잔차 상위 10%)에서 신호 평균 차이 (설명용)
    hi = M.res >= M.res.quantile(.9)
    out[c]['hi_vs_rest_median'] = [round(float(M.loc[hi, c].median()), 3), round(float(M.loc[~hi, c].median()), 3)]
# 우연 대조: 무작위 잡음 신호 200개의 |rho| 95% 분위
null = []
for _ in range(200):
    M['_z'] = rng.normal(size=len(M)); null.append(abs(perm_p(M, '_z', B=1)[0]))
out['null_abs_rho_q95'] = round(float(np.quantile(null, .95)), 4)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'indoor_signals_v1_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
