# -*- coding: utf-8 -*-
"""AG1 (2026-10-12 집 클로드; 사용자: '경과일 추정하는 방법 실험').  1단계 서술(신호 확인).
규정 안 경과일 추정 age_hat(b): b 의 달력 날짜 d(DM11)에서 하루씩 거슬러 k = 1, 2, ... 날짜 d−k 에 '허용 기록' 중 고EC 지문 기록이 있으면 계속,
  허용 기록이 있는데 지문 기록이 없으면 멈춤, 허용 기록이 하나도 없는 날(지워진 날 등)은 건너뜀(연속 2일까지).
  허용 기록 = 학습 기록(모든 ID) + b 와 같은 ID·앞 번호 평가 기록 (입력만 사용; 정답 안 씀).  b 자신은 제외.
  고EC 지문(하루 전체, 앞선 날 기록이라 전부 사용 가능): 커튼 8시 0·10시 ≥ 95·17시 0, CO2 공급 8~14시 최대 < 30, 유동팬 하루 평균 < 5, 실내 CO2 하루 평균 ≥ 390 (HS2·HS3).
보고: 학습 고EC 기록(하루 평균 ≥ 1)에서 age_hat vs 진짜 경과일(진단용 연결, HS4) 상관, age_hat 과 모델 잔차(DIAG10 OOF) 관계(구간 평균),
  보통 기록 중 b 자신이 지문 고EC 인 기록의 age_hat 분포(오탐 정도), 평가 고EC 지문 기록의 age_hat.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
pv = lambda v: X.pivot(index="rec", columns="h", values=v)
sh, co = pv("act_shade"), pv("act_co2")
fp = ((sh[8] == 0) & (sh[10] >= 95) & (sh[17] == 0) & (co.loc[:, 8:14].max(1) < 30) & (X.groupby("rec").act_circfan.mean() < 5) & (X.groupby("rec").in_co2.mean() >= 390))
CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day)); SPL = dict(zip(CAL.rec, CAL.split))
byday = CAL.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
def allowed(b, x):
    if x == b: return False
    if SPL.get(x) == "train": return True
    return x[:3] == b[:3] and int(x[4:7]) < int(b[4:7])
def age_hat(b):
    d = DAY.get(b)
    if d is None or not np.isfinite(d): return np.nan
    k, gaps, age = 1, 0, 0
    while k < 40:
        al = [x for x in byday.get(d - k, []) if allowed(b, x)]
        if not al:
            gaps += 1
            if gaps > 2: break
        else:
            if any(fp.get(x, False) for x in al): age = k; gaps = 0
            else: break
        k += 1
    return age
y = pd.Series({rid(i): float(np.nanmean(EC[i])) for i in np.where(TRAIN)[0] if FARM[i] in ("F13", "F47")})
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
def age_true(r):
    k, x = 0, r
    while True:
        p = pre.get(x)
        if p is None or p not in y.index or y[p] < 1: return k
        k += 1; x = p
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
q = O.groupby("rec").sg.mean()
T = pd.DataFrame([dict(rec=r, y=y[r], fp=bool(fp.get(r, False)), age_hat=age_hat(r), age_true=age_true(r) if y[r] >= 1 else np.nan, r=y[r] - q[r] if r in q.index else np.nan) for r in y.index])
H = T[T.y >= 1]
pd.set_option("display.width", 200)
m = H.age_hat.notna()
print(f"high-EC training records {len(H)} (own fingerprint high {int(H.fp.sum())}) | corr(age_hat, age_true) {np.corrcoef(H.age_hat[m], H.age_true[m])[0, 1]:.2f}")
print("   residual by age_hat band (high-EC):", H.groupby(pd.cut(H.age_hat, [-1, 0, 2, 5, 40])).r.agg(["count", "mean"]).round(2).to_dict("index"))
F = T[T.fp]
print(f"\nrecords whose OWN fingerprint is high: {len(F)} | high-EC {int((F.y >= 1).sum())}")
print("   y and residual by age_hat band:", F.groupby(pd.cut(F.age_hat, [-1, 0, 2, 5, 40])).agg(n=("y", "size"), y=("y", "mean"), high_rate=("y", lambda s: (s >= 1).mean()), resid=("r", "mean")).round(2).to_dict("index"))
print(H.sort_values("age_true")[["rec", "y", "fp", "age_true", "age_hat", "r"]].round(2).to_string(index=False))
ev = [r for r in fp.index if fp[r] and SPL.get(r) == "eval"]
P = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); P["rec"] = P.row_id.str[:7]; pm = P.groupby("rec").sub_ec.mean()
print("\neval records with high fingerprint:", [(r, DAY.get(r), age_hat(r), round(pm[r], 2)) for r in sorted(ev, key=lambda r: DAY.get(r, 0))])
