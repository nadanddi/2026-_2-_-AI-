# -*- coding: utf-8 -*-
"""HX9-OPS v1 (2026-10-13 집 클로드) — 가족: 관리·구동기 지문 (management / actuator signatures).
사용자 요청: '고EC 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인'.  이 스크립트 = 파생 변수 만들기만 (판정은 ops_test_v1.py).

========================  계산 전에 적어 둔 예측 (사전 기록, 수정 금지)  ========================
기제 가설: EC 수준은 재배자 결정(양액 처방·급액량·배액/세척 flush)이 정하고, 같은 사람이 같은 날 하는 구동기 운영
(CO2 공급 체제, 보온·차광 커튼 일정, 포그, 난방 일정, 환기 정책, 그리고 전날 대비 운영 지문의 갑작스러운 변화)에 흔적이 남는다.
(일반 농학 지식 표시: 겨울 밀폐·CO2 시비·낮은 급액은 함께 가는 경우가 많다 — 이는 일반 지식이지 이 자료에서 읽은 사실이 아님.)

이 기제가 '고EC 크기'를 움직인다면 보여야 할 흔적:
  P1. 고EC 기록(hi10) 안에서 어떤 운영 변수가 하루 평균 EC(Y1)뿐 아니라 모델 잔차(Y2 = y − sg)와도 |r| ≥ .35, 사슬 부트스트랩 CI 가
      0 을 제외, sg 기준선에 더했을 때 사슬 단위 교차적합 R2 이득 > 0.  (모델은 이미 시간별 원 구동기를 입력으로 쓰므로,
      '일정·체제·변화' 형태로 다시 짠 변수가 원 입력이 못 담은 정보를 줘야 함.)
  P2. 같은 부호가 F13·F47 양쪽, 1차·2차 양쪽(2차 고EC 는 5기록뿐 → 부호만)에서.
  P3. 일정 변화 점수(전날 대비 운영 지문 거리)가 고EC 에피소드 안 하루 EC 변화(Y3) 크기와 이어지고, 특히 하락(flush)일에 크게 나타남.
  P4. 고EC 에피소드 진입일(전날 보통 → 오늘 고EC)에 운영 지문 변화가 다른 날보다 큼.
  P5. 한 시간 EC 계단 하락(flush 추정) 시각에 구동기 흔적(포그 펄스, 환기·커튼·난방 변화)이 같은 시각 위약보다 뚜렷.
보이면 안 되는 것 (보이면 '크기 기제' 아님):
  N1. Y1 과는 연관되나 Y2 와는 ≈0 → 이미 모델이 쓰고 있거나 '고EC 여부'만 알려 주는 표지.
  N2. 한 사슬(특히 F47 89~98일 10기록 사슬)을 빼면 부호가 바뀜.
  N3. 계절(하루 평균 외기온) 통제 후 사라짐.
  N4. 농장·차수 사이 부호 반전.
사전 기대(솔직히): 이전 6.690/6.691(지문은 '누가'만, 크기 아님), 6.464/6.470(커튼 일정 확인 실패), 6.794(운영 편차 실패),
  6.653/6.659(계단 시각 구동기 흔적 없음)를 보면 대부분 none/weak 로 예상. Y1 연관은 사슬 간 차이(계절·동)로 보일 수 있음.
판정 규칙(사전 고정, ops_test_v1.py 에 코드로): signal = |r_resid(high)| ≥ .35 AND 사슬 부트스트랩 95% CI 가 0 제외 AND 사슬 LOCO
  R2 이득 > 0; weak = 그중 하나라도 만족 (|r| ≥ .25 또는 CI 0 제외 또는 이득 > 0) 하지만 signal 아님; none = 나머지.
================================================================================================

평가 행 사용 가능성 (rule: 평가 행 h 에서 그 기록 0..h 입력, 같은 온실 더 이른 번호 기록 입력, 학습 입력·정답만):
  avail_h = 이 변수가 확정되는 가장 이른 시각. 23 이면 23시 행에서만 확정(그 전에는 부분값만). 'censored' = 관측될 때 확정.
  전날 링크: clean(정답 연속성 + DM11) = diagnostic, input 링커(입력 연속성만, visible 후보) = usable.
산출: hx9/ops/vars_v1.csv (rec 키), hx9/ops/vars_v1_meta.csv (정의·usable·avail_h), hx9/ops/link_input_v1.csv
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
     "act_heating", "act_circfan", "act_co2", "act_fog"]
ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]

# ---------------- all records (train + eval) hourly inputs ----------------
def load(fn, split):
    D = pd.read_csv(os.path.join(env.DATA, fn))
    D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["hour"] = D.row_id.str[8:].astype(int); D["split"] = split
    return D[["rec", "hour", "split"] + V]
A = pd.concat([load("train_X.csv", "train"), load("test_X.csv", "eval")], ignore_index=True)
recs = sorted(A.rec.unique()); NR = len(recs); RI = {r: k for k, r in enumerate(recs)}
assert NR == 460, NR
ARR = {v: A.pivot(index="rec", columns="hour", values=v).loc[recs].values.astype(float) for v in V}
FARMa = np.array([r[:3] for r in recs]); NUMa = np.array([int(r[4:]) for r in recs])
SPLIT = A.groupby("rec").split.first().loc[recs].values; TRN = SPLIT == "train"
print("records", NR, "train", TRN.sum(), "eval", (~TRN).sum())

# ---------------- input-only predecessor linker (usable) ----------------
OUT = np.stack([ARR[v] for v in V[:4]], 2)               # [NR,24,4]
INA = np.stack([ARR[v] for v in ["in_temp", "in_hum", "in_co2", "act_thermal", "act_shade", "act_heating", "act_circfan", "act_vent"]], 2)
sdO = np.nanstd(OUT[TRN].reshape(-1, 4), 0)
sdI = np.nanstd(np.diff(INA[TRN], axis=1).reshape(-1, INA.shape[2]), 0) + 1e-6
Of = OUT.reshape(NR, -1)
def same_date(i):
    return (np.abs(Of - Of[i]) < 1e-6).mean(1) >= 0.7
def input_pred(i):
    f, n = FARMa[i], NUMa[i]
    vis = ((FARMa == f) & (NUMa < n)) | ((FARMa != f) & TRN)
    vis[i] = False
    vis &= ~same_date(i)
    cand = np.where(vis)[0]
    pa = OUT[cand, 23] + 0.5 * (OUT[cand, 23] - OUT[cand, 22])
    oj = np.sum(np.array([1, 1, .3, .5]) * np.abs(pa - OUT[i, 0]) / sdO, 1)
    dv = np.minimum(np.nan_to_num(np.abs(INA[cand, 23] - INA[i, 0]) / sdI, nan=2.0), 6).sum(1)
    c = oj + 0.5 * dv
    o = np.argsort(c)
    return cand[o[0]], c[o[0]], (c[o[1]] - c[o[0]]) if len(o) > 1 else np.nan

LK = []
for r in R.index:
    i = RI[r]; a, c, mg = input_pred(i)
    LK.append(dict(rec=r, inp_prev=recs[a], inp_cost=c, inp_margin=mg, inp_prev_train=int(TRN[a])))
LK = pd.DataFrame(LK).set_index("rec")
R = R.join(LK)
m = R.prev_rec.notna()
agree = (R.loc[m, "inp_prev"] == R.loc[m, "prev_rec"]).mean()
mh = m & (R.hi10 == 1)
agree_h = (R.loc[mh, "inp_prev"] == R.loc[mh, "prev_rec"]).mean()
print(f"input linker vs clean predecessor: agree {agree:.3f} (n {m.sum()}); among hi10 {agree_h:.3f} (n {mh.sum()})")
LK.join(R[["prev_rec"]]).to_csv(os.path.join(HERE, "link_input_v1.csv"), encoding="utf-8")

# ---------------- helpers ----------------
def first_cross_up(x, lo, hi, thr=50):
    for h in range(max(lo, 1), hi + 1):
        if x[h] >= thr and x[h - 1] < thr: return float(h)
    return np.nan
def first_cross_down(x, lo, hi, thr=50):
    for h in range(max(lo, 1), hi + 1):
        if x[h] < thr and x[h - 1] >= thr: return float(h)
    return np.nan
def n_runs(on):
    on = np.asarray(on, bool); return int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
def first_on(x):
    w = np.where(x > 0)[0]; return float(w[0]) if len(w) else 24.0

FP = np.stack([ARR[a] for a in ACT], 2) / 100.0          # [NR,24,7] fingerprint (0..1 scale)
def fp_L1(i, j, hours=slice(0, 24)):
    return float(np.nanmean(np.abs(FP[i, hours] - FP[j, hours])))

rows = []
for r in R.index:
    i = RI[r]
    ve, sh, th, he, fa, co, fo = (ARR[a][i] for a in ACT)
    d = dict(rec=r)
    # A. CO2 dosing regime
    d["co2_hours"] = float((co > 0).sum()); d["co2_sum"] = float(co.sum()); d["co2_first_h"] = first_on(co)
    d["co2_am_hours"] = float((co[:12] > 0).sum())
    d["co2_offday"] = float(((co[:6] > 0).sum() + (co[18:] > 0).sum()))   # dosing outside 6..17 (atypical)
    # B. thermal curtain schedule
    d["th_open_h"] = first_cross_up(th, 4, 15); d["th_close_h"] = first_cross_down(th, 12, 23)
    d["th_open_dur"] = float((th >= 50).sum()); d["th_trans"] = float((np.abs(np.diff(th)) >= 30).sum())
    d["th_night"] = float(np.mean(th[:6]))
    # C. shade schedule
    d["sh_open_h"] = first_cross_up(sh, 4, 15); d["sh_close_h"] = first_cross_down(sh, 12, 23)
    d["sh_open_dur"] = float((sh >= 50).sum()); d["sh_partial"] = float(((sh > 5) & (sh < 95)).mean())
    d["curtain_decouple"] = float(np.mean(np.abs(sh - th)))
    # D. fog
    d["fog_hours"] = float((fo > 0).sum()); d["fog_pulses"] = float(n_runs(fo > 0)); d["fog_sum"] = float(fo.sum())
    # E. heating schedule
    d["heat_hours"] = float((he > 0).sum()); d["heat_night"] = float(np.mean(he[:6]))
    d["heat_dayfrac"] = float((he[9:17] > 0).mean())
    w = [h for h in range(5, 15) if he[h] > 0]; d["heat_last_am"] = float(max(w)) if w else np.nan
    # F. ventilation policy
    d["vent_first_h"] = first_on(ve); d["vent_hours"] = float((ve > 0).sum()); d["vent_max"] = float(ve.max())
    d["sealed_frac"] = float((ve == 0).mean())
    # G. circulation fan
    d["fan_hours"] = float((fa > 0).sum()); d["fan_trans"] = float(np.sum((fa[1:] > 0) != (fa[:-1] > 0)))
    d["fan_night"] = float(np.mean(fa[:6]))
    # H. activity count
    d["op_n_active"] = float(sum((ARR[a][i] > 0).any() for a in ACT))
    rows.append(d)
F = pd.DataFrame(rows).set_index("rec")

# ---------------- regime clusters / prototypes (inputs only; target encodings cross-fitted by group) ----------------
TRi = np.array([RI[r] for r in R.index])
Z = FP[TRi].reshape(len(TRi), -1)
Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
km = KMeans(n_clusters=6, n_init=20, random_state=0).fit(Zs)
F["op_cluster"] = km.labels_
grp = np.where(R.hi10 == 1, "ep" + R.ep_id.astype(str), np.where(R.fold.notna(), "f" + R.fold.astype(str), "L" + pd.Series(R.index, index=R.index)))
R["grp"] = grp
hi = R.hi10.values == 1; ecm = R.ec_mean.values
te_rate, te_mag, proto, knn = [], [], [], []
for k, r in enumerate(R.index):
    out = R.grp.values != R.grp.values[k]
    cl = F.op_cluster.values == F.op_cluster.values[k]
    s = out & cl
    te_rate.append(hi[s].mean() if s.sum() else np.nan)
    sm = s & hi
    te_mag.append(ecm[sm].mean() if sm.sum() else np.nan)
    ph = Zs[out & hi].mean(0); pn = Zs[out & ~hi].mean(0)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    proto.append(cos(Zs[k], ph) - cos(Zs[k], pn))
    cand = np.where(out & hi)[0]
    dd = np.linalg.norm(Zs[cand] - Zs[k], axis=1)
    knn.append(ecm[cand[np.argsort(dd)[:3]]].mean())
F["te_clu_hirate"] = te_rate; F["te_clu_mag"] = te_mag; F["proto_sim_hi"] = proto; F["knn3_hi_mag"] = knn
ct = pd.crosstab(F.op_cluster, R.hi10)
print("cluster x hi10\n", ct)

# ---------------- schedule change vs predecessor ----------------
def chg(prev_col, tag):
    L1, L1am, ds, dco, dhe, dth, dfa = [], [], [], [], [], [], []
    for r in R.index:
        p = R.at[r, prev_col]
        if not isinstance(p, str):
            L1.append(np.nan); L1am.append(np.nan); ds.append(np.nan); dco.append(np.nan); dhe.append(np.nan); dth.append(np.nan); dfa.append(np.nan); continue
        i, j = RI[r], RI[p]
        L1.append(fp_L1(i, j)); L1am.append(fp_L1(i, j, slice(0, 12)))
        ve_i, ve_j = ARR["act_vent"][i], ARR["act_vent"][j]
        ds.append(float((ve_i == 0).mean() - (ve_j == 0).mean()))
        dco.append(float((ARR["act_co2"][i] > 0).sum() - (ARR["act_co2"][j] > 0).sum()))
        dhe.append(float((ARR["act_heating"][i] > 0).sum() - (ARR["act_heating"][j] > 0).sum()))
        dth.append(float((ARR["act_thermal"][i] >= 50).sum() - (ARR["act_thermal"][j] >= 50).sum()))
        dfa.append(float((ARR["act_circfan"][i] > 0).sum() - (ARR["act_circfan"][j] > 0).sum()))
    F[f"chg_{tag}_L1"] = L1; F[f"chg_{tag}_L1am"] = L1am; F[f"chg_{tag}_sealed"] = ds; F[f"chg_{tag}_co2h"] = dco
    F[f"chg_{tag}_heath"] = dhe; F[f"chg_{tag}_thdur"] = dth; F[f"chg_{tag}_fanh"] = dfa
chg("prev_rec", "clean")
chg("inp_prev", "inp")

F.reset_index().to_csv(os.path.join(HERE, "vars_v1.csv"), index=False, encoding="utf-8")

# ---------------- metadata ----------------
U, D = "usable", "diagnostic only"
meta = {
 "co2_hours": ("CO2 공급(act_co2>0) 시간 수", U, "23"), "co2_sum": ("act_co2 하루 합", U, "23"),
 "co2_first_h": ("처음 CO2 공급 시각(없으면 24)", U, "censored"), "co2_am_hours": ("0~11시 CO2 공급 시간 수", U, "11"),
 "co2_offday": ("6~17시 밖 CO2 공급 시간 수(비정형 공급)", U, "23"),
 "th_open_h": ("보온커튼 act_thermal 50 상향 교차 시각(4~15시)", U, "censored(≤15)"),
 "th_close_h": ("보온커튼 50 하향 교차 시각(12~23시)", U, "censored(≤23)"),
 "th_open_dur": ("act_thermal ≥ 50 시간 수", U, "23"), "th_trans": ("보온커튼 |Δ| ≥ 30 전환 수", U, "23"),
 "th_night": ("0~5시 보온커튼 평균", U, "5"),
 "sh_open_h": ("차광 act_shade 50 상향 교차 시각(4~15시)", U, "censored(≤15)"),
 "sh_close_h": ("차광 50 하향 교차 시각(12~23시)", U, "censored(≤23)"),
 "sh_open_dur": ("act_shade ≥ 50 시간 수", U, "23"), "sh_partial": ("차광 5~95 중간 위치 시간 비율", U, "23"),
 "curtain_decouple": ("|차광 − 보온| 하루 평균(두 커튼 따로 움직임)", U, "23"),
 "fog_hours": ("포그 가동 시간 수", U, "23"), "fog_pulses": ("포그 가동 덩어리 수", U, "23"), "fog_sum": ("포그 하루 합", U, "23"),
 "heat_hours": ("난방 가동 시간 수", U, "23"), "heat_night": ("0~5시 난방 평균", U, "5"),
 "heat_dayfrac": ("9~16시 난방 가동 비율", U, "16"), "heat_last_am": ("5~14시 중 마지막 난방 가동 시각", U, "14"),
 "vent_first_h": ("처음 환기 개방 시각(없으면 24)", U, "censored"), "vent_hours": ("환기 개방 시간 수", U, "23"),
 "vent_max": ("환기 하루 최대", U, "23"), "sealed_frac": ("환기 0 비율(밀폐; 6.11 기준 변수)", U, "23"),
 "fan_hours": ("유동팬 가동 시간 수", U, "23"), "fan_trans": ("유동팬 켜짐/꺼짐 전환 수", U, "23"),
 "fan_night": ("0~5시 유동팬 평균", U, "5"), "op_n_active": ("하루 한 번이라도 움직인 구동기 수(0~7)", U, "23"),
 "op_cluster": ("KMeans(6) 24h×7 구동기 지문 군집(학습 입력만)", U, "23"),
 "te_clu_hirate": ("같은 군집 학습 기록의 hi10 비율(자기 그룹 제외 교차적합)", U, "23"),
 "te_clu_mag": ("같은 군집 hi10 학습 기록의 EC 평균(자기 그룹 제외)", U, "23"),
 "proto_sim_hi": ("고EC 지문 원형 코사인 − 보통 원형 코사인(자기 그룹 제외)", U, "23"),
 "knn3_hi_mag": ("지문 최근접 hi10 학습 기록 3개의 EC 평균(자기 그룹 제외)", U, "23"),
 "chg_clean_L1": ("clean 전날 대비 24h×7 구동기 지문 평균 절대차", D, "23"),
 "chg_clean_L1am": ("clean 전날 대비 0~11시 지문 차", D, "11"),
 "chg_clean_sealed": ("밀폐 비율 − clean 전날", D, "23"), "chg_clean_co2h": ("CO2 공급 시간 − clean 전날", D, "23"),
 "chg_clean_heath": ("난방 시간 − clean 전날", D, "23"), "chg_clean_thdur": ("보온 ≥50 시간 − clean 전날", D, "23"),
 "chg_clean_fanh": ("유동팬 시간 − clean 전날", D, "23"),
 "chg_inp_L1": ("입력 링커 전날 대비 24h×7 지문 차", U, "23"), "chg_inp_L1am": ("입력 링커 전날 대비 0~11시 지문 차", U, "11"),
 "chg_inp_sealed": ("밀폐 비율 − 입력 링커 전날", U, "23"), "chg_inp_co2h": ("CO2 공급 시간 − 입력 링커 전날", U, "23"),
 "chg_inp_heath": ("난방 시간 − 입력 링커 전날", U, "23"), "chg_inp_thdur": ("보온 ≥50 시간 − 입력 링커 전날", U, "23"),
 "chg_inp_fanh": ("유동팬 시간 − 입력 링커 전날", U, "23"),
}
M = pd.DataFrame([dict(variable=k, definition=v[0], usable=v[1], avail_h=v[2]) for k, v in meta.items()])
assert set(M.variable) == set(F.columns), set(F.columns) ^ set(M.variable)
M.to_csv(os.path.join(HERE, "vars_v1_meta.csv"), index=False, encoding="utf-8")
print("variables", len(M), "| usable", (M.usable == U).sum(), "diagnostic", (M.usable == D).sum())
print(F.describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())
