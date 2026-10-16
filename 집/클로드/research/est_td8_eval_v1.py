# EST (2026-10-13 집 클로드; 사용자: 'TD8 제출하면 온도 몇쯤?') 서술: TD8 규칙을 평가 60 에 적용했을 때 바뀌는 기록·행과 RMS 변화, 점수 범위 산수.
import os, sys, importlib.util
import env
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])
PEV = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_eval.csv"))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
NEW, ng, nga = new_anchors(PEV, lambda r: TRAIN.copy(), v7e)
print(f"eval: v7 anchors {len(v7e)} | new v12 anchors {len(NEW)} (gated {ng}, HP2-agreed {nga})")
for r, p in sorted(NEW.items()):
    row = PEV[PEV.rec == r].iloc[0]; print(f"   {r} <- {p} p {row.p:.2f} dT {row.dT:+.1f} date {row.date_status}")
S = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
new = S.sub_temp.values.copy()
for r, p in NEW.items():
    m = (S.rec == r).values; hh = S.h.values[m]; q0 = S.sub_temp.values[m][hh == 0][0]
    new[m] = S.sub_temp.values[m] + A7.BETA[hh] * np.clip(ST[idx[p], 23] - .36 - q0, -2, 2)
d = new - S.sub_temp.values; r_ = float(np.sqrt(np.mean(d ** 2))); s0 = .4883
print(f"rows changed {int((np.abs(d) > 1e-9).sum())} | RMS change {r_:.4f} | mean change {d.mean():+.4f}")
for rho in (-.3, -.225, -.1, 0, .1):
    print(f"   corr(change, current error) {rho:+.3f} -> temp {np.sqrt(s0**2 + r_**2 + 2*rho*r_*s0):.4f}")
print(f"   arithmetic range: {np.sqrt(max(0, s0**2 + r_**2 - 2*r_*s0)):.4f} ~ {np.sqrt(s0**2 + r_**2 + 2*r_*s0):.4f}")
