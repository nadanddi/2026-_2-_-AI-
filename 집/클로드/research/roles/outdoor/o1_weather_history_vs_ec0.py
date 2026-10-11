# -*- coding: utf-8 -*-
"""실외 날씨 이력(지난 1~7일 일사·흐림 연속·한파) vs EC 0시 잔차 (CUR, DIAG10).
달력: 1차 구간(day<179) 학습 기록의 외기를 날짜(season)별로 평균 -> 2차 구간은 season표로 매핑.
우연 대조: 특징 달력을 k일(20~150) 원형 이동한 상관의 분포."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from scipy.stats import spearmanr

X = pd.read_csv(os.path.join(env.DATA, 'train_X.csv'))
X['farm'] = X.row_id.str[:3]; X['day'] = X.row_id.str[4:7].astype(int); X['hour'] = X.row_id.str[-2:].astype(int)
S = pd.read_csv(r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC\training_day_to_season.csv")

# 일별 외기 요약
D = X.groupby(['farm', 'day']).agg(rad=('out_rad', 'sum'), tmean=('out_temp', 'mean'),
                                   tmin=('out_temp', 'min'), hum=('out_hum', 'mean'),
                                   wsp=('out_wspd', 'mean'), nrad=('out_rad', 'count')).reset_index()
D = D.merge(S, on=['farm', 'day'], how='left')
D['phase'] = np.where(D.day >= 179, 2, 1)
# 달력: 1차 구간, season 정수, 두 온실 평균
cal = D[D.phase == 1].groupby('season')[['rad', 'tmean', 'tmin', 'hum', 'wsp']].mean()
cal = cal.reindex(np.arange(1, 179)).interpolate(limit_direction='both')
print('달력 일수', len(cal), ' 1·2온실 일사 상관',
      D[D.phase == 1].pivot_table(index='season', columns='farm', values='rad').corr().iloc[0, 1].round(3))
q = cal.rad.quantile(0.33)

def feats(c):
    f = pd.DataFrame(index=c.index)
    f['rad_d1'] = c.rad.shift(1)
    f['rad_3'] = c.rad.shift(1).rolling(3).sum()
    f['rad_7'] = c.rad.shift(1).rolling(7).sum()
    cloudy = (c.rad < q).astype(int)
    st = []; n = 0
    for v in cloudy.shift(1).fillna(0):
        n = n + 1 if v else 0; st.append(n)
    f['cloudy_streak'] = st
    f['tmin_3'] = c.tmin.shift(1).rolling(3).min()
    f['hum_3'] = c.hum.shift(1).rolling(3).mean()
    # 계절 추세 제거판(지난 3일 - 지난 30일 평균)
    f['rad_3_anom'] = c.rad.shift(1).rolling(3).mean() - c.rad.shift(1).rolling(30, min_periods=10).mean()
    f['rad_7_anom'] = c.rad.shift(1).rolling(7).mean() - c.rad.shift(1).rolling(30, min_periods=10).mean()
    f['tmin_3_anom'] = c.tmin.shift(1).rolling(3).mean() - c.tmin.shift(1).rolling(30, min_periods=10).mean()
    return f
F = feats(cal)
names = list(F.columns)

# 0시 잔차
R = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
R = R[(R.tag == 'CUR') & (R.validator == 'DIAG10') & (R.hour == 0)]
R['res'] = R.sub_ec - R.sg
R = R.groupby(['farm', 'day']).agg(res=('res', 'mean'), y=('sub_ec', 'first')).reset_index()
R = R.merge(D[['farm', 'day', 'season', 'phase', 'rad']], on=['farm', 'day'])
R['s'] = R.season.round().clip(1, 178).astype(int)
print('0시 잔차 행', len(R), R.groupby(['phase', 'farm']).size().to_dict(), ' |res| 평균', R.res.abs().mean().round(4))

rng = np.random.default_rng(0)
out = []
for (ph, fm), g in list(R.groupby(['phase', 'farm'])) + [(('all', 'all'), R)]:
    for nm in names:
        x = F[nm].reindex(g.s).values; ok = ~np.isnan(x)
        if ok.sum() < 15: continue
        rho = spearmanr(x[ok], g.res.values[ok])[0]
        rho_y = spearmanr(x[ok], g.y.values[ok])[0]
        null = []
        for k in range(20, 151, 3):
            xs = F[nm].values; xs = np.roll(xs, k)
            xk = pd.Series(xs, index=F.index).reindex(g.s).values; ok2 = ~np.isnan(xk)
            null.append(spearmanr(xk[ok2], g.res.values[ok2])[0])
        null = np.abs(np.array(null))
        p = (1 + (null >= abs(rho)).sum()) / (1 + len(null))
        out.append(dict(phase=ph, farm=fm, feat=nm, n=int(ok.sum()), rho_res=round(rho, 3),
                        rho_y=round(rho_y, 3), null95=round(np.quantile(null, .95), 3), p_shift=round(p, 3)))
O = pd.DataFrame(out)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300)
print(O.to_string())
O.to_csv(r"C:\work\farmai\집\클로드\research\roles\outdoor\o1_result.csv", index=False, encoding='utf-8-sig')

# 2차 구간: 계절 지표가 보간된(같은 season 반복) 날 비율
g2 = R[R.phase == 2]
print('2차 season 비정수 비율', (g2.season % 1 != 0).mean().round(3), ' season 중복 비율', g2.duplicated(['farm', 's']).mean().round(3))
