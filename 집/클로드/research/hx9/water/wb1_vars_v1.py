# -*- coding: utf-8 -*-
"""HX9 / water  WB1 — 물 수지·증산 가족 파생 변수 만들기 (2026-10-13 집 클로드, 하위 작업).
사용자 요청: '고EC 의 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인. 파생 변수 직접 만들어 확인해도 됨.'
이 스크립트는 변수만 만든다 (판정은 wb2_tests_v1.py). 예측(지문)은 wb2 머리말에 먼저 적었다.

메커니즘 (일반 원예 지식, 우리 자료에서 확인한 것 아님):
  배지 재배에서 급액·배액 체제가 고정이면, 증산이 적은 날(흐림·고습·밀폐) → 배액·세척이 줄어 염류가 쌓여 EC 상승;
  일사·VPD 가 크면 급액이 늘어 세척 → EC 하락. 반대로 급액이 증산을 못 따라가면 맑은 날 농축 → EC 상승도 가능.
  즉 부호는 사전에 두 갈래 (A 세척 부족형: 증산↓ → EC↑ / B 농축형: 증산↑ → EC↑).
  우리 자료에는 급액·배액·배지 무게·실내 일사(in_rad)·펌프 관측이 없다 (F13/F47 은 act_pump/valve/in_rad 모두 NaN) → 전부 대리값.

변수 정의 (학습 기록 400개, 기록 = 24시간 하루). 학습 입력의 결측은 기록 안 선형 보간(학습 입력 보수는 비인과 허용, 10-10 규정).
  물리식(일반 지식): es(T)=0.6108·exp(17.27T/(T+237.3)) kPa, VPD=es(Tin)(1-RH/100), AH=2165·e/(T+273.15) g/m3, 이슬점 Td.
  실외 일사 바닥값 5 W/m2 를 빼고 0 으로 자름.  실내 일사 대리 radIn = rad·0.65·(1-0.6·shade/100)·(1-0.4·thermal/100)
  (투과율 0.65·차광 60%·보온커튼 40% 차단은 가정값, 근거 없음 → 원 일사 합도 같이 둠).
  증산 대리 ET (Baille 류 단순식, 일반 지식): E[W/m2] = 0.30·radIn + 20·VPD(낮, rad>10) / 10·VPD(밤); 하루 합 / λ(680 Wh/kg) → mm.
  낮 = 09~16시, 밤 = 20~23 + 0~5시 (hx9 공용표와 같음).

평가 행 사용 가능성 (열 사전 vars_v1_columns.csv 에 적음):
  U   : 평가 기록 자기 0..h 입력만으로 계산 가능 (누적형은 0..h 부분합이 그 시각의 사용판; 하루 전체값은 h=23 에서만).
  U5  : 0~5시 창 → h>=5 부터 완전.
  Uid : 같은 ID 앞 기록(num-2) 입력 → 허용 (같은 ID 이전 기록 입력). 단 그 기록이 같은 물리 동의 전날인지는 보장 없음.
  Ucal: 학습 달력 위치가 필요 (평가 기록 날짜는 자기 0..h 실외를 학습 쌍둥이와 맞춰 추정, 0시 일치 375/400) — 원칙상 가능, 여기 값은 DM11 로 계산 → 서술용.
  D   : truth.pkl clean 링크(대상 자신 정답으로 고른 전날) 또는 에피소드(정답) 필요 → 진단 전용.
산출: hx9/water/vars_v1.csv (rec 키), hx9/water/vars_v1_columns.csv, hx9/water/vars_running_v1.csv (h=5,11,17,23 의 0..h 인과판, ffill).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

R = pd.read_csv(os.path.join(DATA, "records.csv")).set_index("rec")
H = pd.read_csv(os.path.join(DATA, "hourly.csv")).sort_values(["rec", "hour"])
V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
     "act_heating", "act_circfan", "act_co2", "act_fog"]


def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def ahum(T, RH): return 2165.0 * (np.clip(RH, 1, 100) / 100.0) * es(T) / (T + 273.15)
def dewpt(T, RH):
    e = np.clip(RH, 1, 100) / 100.0 * es(T); g = np.log(e / 0.6108); return 237.3 * g / (17.27 - g)
def nmean(x):
    x = np.asarray(x, float); return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
def nsum(x):
    x = np.asarray(x, float); return float(np.nansum(x)) if np.isfinite(x).any() else np.nan

HRS = np.arange(24)
DAY = (HRS >= 9) & (HRS <= 16)
NIGHT = (HRS <= 5) | (HRS >= 20)
LAM = 680.0  # Wh/kg


def same_day_feats(a, h=23):
    """a: dict var -> 24 array.  uses hours 0..h only.  returns dict."""
    m = HRS <= h
    T, RH, To, RHo = a["in_temp"], a["in_hum"], a["out_temp"], a["out_hum"]
    rad = np.clip(a["out_rad"] - 5.0, 0, None)
    shade, therm, vent, heat, fog = a["act_shade"], a["act_thermal"], a["act_vent"], a["act_heating"], a["act_fog"]
    radIn = rad * 0.65 * (1 - 0.6 * shade / 100.0) * (1 - 0.4 * therm / 100.0)
    vpd = es(T) * (1 - np.clip(RH, 0, 100) / 100.0)
    AHi, AHo = ahum(T, RH), ahum(To, RHo)
    Td = dewpt(T, RH)
    isday = rad > 10
    E = np.where(isday, 0.30 * radIn + 20.0 * vpd, 10.0 * vpd)  # W/m2
    dAH = np.r_[np.nan, np.diff(AHi)]
    f = {}
    f["rad_out_sum"] = nsum(rad[m]) * 3600 / 1e6               # MJ/m2
    f["rad_in_sum"] = nsum(radIn[m]) * 3600 / 1e6
    f["shade_loss"] = 1 - f["rad_in_sum"] / (0.65 * f["rad_out_sum"]) if f["rad_out_sum"] > 0.05 else np.nan
    f["vpd_day"] = nmean(vpd[m & DAY]) if (m & DAY).any() else np.nan
    f["vpd_24"] = nmean(vpd[m])
    f["vpd_rad"] = nsum((vpd * radIn)[m]) / 1e3
    f["et_day"] = nsum(np.where(isday, E, 0)[m]) / LAM
    f["et_night"] = nsum(np.where(~isday, E, 0)[m]) / LAM
    f["et_total"] = nsum(E[m]) / LAM
    f["et_night_share"] = f["et_night"] / f["et_total"] if f["et_total"] > 0 else np.nan
    f["vpd_night"] = nmean(vpd[m & NIGHT]) if (m & NIGHT).any() else np.nan
    f["hum_night"] = nmean(RH[m & NIGHT]) if (m & NIGHT).any() else np.nan
    f["cond_hours"] = float(np.nansum(((T - Td) < 1.0)[m]))
    f["dew_margin_min"] = float(np.nanmin((T - Td)[m])) if np.isfinite((T - Td)[m]).any() else np.nan
    f["vent_vpd"] = nsum((vent / 100.0 * vpd)[m])
    f["vent_ahx"] = nsum((vent / 100.0 * np.clip(AHi - AHo, 0, None))[m])
    f["sealed_hours"] = float(np.nansum((vent < 1)[m]))
    f["sealed_day_hours"] = float(np.nansum((vent < 1)[m & DAY]))
    f["heat_deg_h"] = nsum(np.clip(T - To, 0, None)[m])
    f["heating_sum"] = nsum(heat[m]) / 100.0
    sr = np.where((rad >= 50) & m)[0]
    f["sunrise_hour"] = float(sr[0]) if len(sr) else np.nan
    w = m & (HRS >= 8) & (HRS <= 10)
    f["morning_ramp"] = nmean(rad[w]) if w.any() else np.nan
    if h >= 7:
        w2 = m & (HRS >= 7) & (HRS <= 13)
        f["ah_rise_morning"] = float(np.nanmax(AHi[w2]) - AHi[6])
        f["hum_drop_morning"] = float(RH[6] - np.nanmin(RH[w2]))
    else:
        f["ah_rise_morning"] = np.nan; f["hum_drop_morning"] = np.nan
    mm = m & DAY & (vent < 5) & (fog <= 0)
    f["dah_closed_day"] = nmean(dAH[mm]) if mm.sum() >= 2 else np.nan
    f["ahx_day"] = nmean((AHi - AHo)[m & DAY]) if (m & DAY).any() else np.nan
    f["tdiff_day"] = nmean((T - To)[m & DAY]) if (m & DAY).any() else np.nan
    w5 = HRS <= min(h, 5)
    f["vpd_0_5"] = nmean(vpd[w5]); f["hum_0_5"] = nmean(RH[w5])
    f["cond_0_5"] = float(np.nansum(((T - Td) < 1.0)[w5])); f["sealed_0_5"] = float(np.nansum((vent < 1)[w5]))
    f["heat_deg_0_5"] = nsum(np.clip(T - To, 0, None)[w5])
    return f


SAME = ["rad_out_sum", "rad_in_sum", "shade_loss", "vpd_day", "vpd_24", "vpd_rad", "et_day", "et_night", "et_total", "et_night_share",
        "vpd_night", "hum_night", "cond_hours", "dew_margin_min", "vent_vpd", "vent_ahx", "sealed_hours", "sealed_day_hours",
        "heat_deg_h", "heating_sum", "sunrise_hour", "morning_ramp", "ah_rise_morning", "hum_drop_morning", "dah_closed_day",
        "ahx_day", "tdiff_day", "vpd_0_5", "hum_0_5", "cond_0_5", "sealed_0_5", "heat_deg_0_5"]

arr_int, arr_ff = {}, {}
for rec, g in H.groupby("rec"):
    g = g.set_index("hour").reindex(range(24))[V]
    arr_int[rec] = {v: g[v].interpolate(limit_direction="both").values.astype(float) for v in V}
    arr_ff[rec] = {v: g[v].ffill().bfill().values.astype(float) for v in V}   # bfill only matters if hour 0 is NaN

F = pd.DataFrame({rec: same_day_feats(a, 23) for rec, a in arr_int.items()}).T.loc[R.index]
F.index.name = "rec"

# running (causal 0..h, ffill) versions for hourly check
run = []
for rec, a in arr_ff.items():
    for h in (5, 11, 17, 23):
        d = same_day_feats(a, h); d.update(rec=rec, h=h); run.append(d)
RUN = pd.DataFrame(run)

# ---------------- lagged: clean predecessor (D), linked chain back 3 (D), same-ID num-2 (Uid) ----------------
prev = R.prev_rec.to_dict()
def back_chain(r, k):
    out = []; cur = r
    for _ in range(k):
        p = prev.get(cur)
        if not isinstance(p, str): break
        out.append(p); cur = p
    return out
F["prev_et_total"] = [F.at[prev[r], "et_total"] if isinstance(prev[r], str) else np.nan for r in F.index]
F["prev_rad_in_sum"] = [F.at[prev[r], "rad_in_sum"] if isinstance(prev[r], str) else np.nan for r in F.index]
F["prev_sealed_hours"] = [F.at[prev[r], "sealed_hours"] if isinstance(prev[r], str) else np.nan for r in F.index]
F["d_et_total"] = F.et_total - F.prev_et_total
F["et_3link"] = [nmean([F.at[p, "et_total"] for p in back_chain(r, 3)]) if back_chain(r, 3) else np.nan for r in F.index]
F["sealed_3link"] = [nmean([F.at[p, "sealed_hours"] for p in back_chain(r, 3)]) if back_chain(r, 3) else np.nan for r in F.index]
F["vpd_night_3link"] = [nmean([F.at[p, "vpd_night"] for p in back_chain(r, 3)]) if back_chain(r, 3) else np.nan for r in F.index]
lag2 = {r: f"{r[:3]}_{int(r[4:]) - 2:03d}" for r in F.index}
F["lag2id_et_total"] = [F.at[lag2[r], "et_total"] if lag2[r] in F.index else np.nan for r in F.index]
F["lag2id_sealed_hours"] = [F.at[lag2[r], "sealed_hours"] if lag2[r] in F.index else np.nan for r in F.index]

# ---------------- multi-day outdoor (calendar).  Only TRAINING records' outdoor is used (test_X masked). ----------------
day11 = R.dm11_day
dd = pd.DataFrame({"day": day11, "rad": F.rad_out_sum}).dropna().groupby("day").rad.mean()
def win(d, lo, hi):
    if not np.isfinite(d): return np.nan
    v = [dd[x] for x in range(int(d) - hi, int(d) - lo + 1) if x in dd.index]
    return float(np.mean(v)) if v else np.nan
F["rad_out_3d"] = [win(d, 1, 3) for d in day11]
F["rad_out_7d"] = [win(d, 1, 7) for d in day11]
env_max = {d: max([dd[x] for x in range(int(d) - 7, int(d) + 8) if x in dd.index]) for d in dd.index}
F["cloud_idx"] = [F.at[r, "rad_out_sum"] / env_max[d] if np.isfinite(d) and d in env_max and env_max[d] > 0 else np.nan
                  for r, d in zip(F.index, day11)]
cid = {d: dd[d] / env_max[d] for d in dd.index}
F["dull_7d"] = [float(sum(cid[x] < 0.5 for x in range(int(d) - 7, int(d)) if x in cid)) if np.isfinite(d) else np.nan for d in day11]
# DMT3e training-only calendar version (seg, day) -> usable in principle
segday = R[["dmt_seg", "dmt_day"]]
dmt = pd.DataFrame({"seg": R.dmt_seg, "day": R.dmt_day, "rad": F.rad_out_sum}).groupby(["seg", "day"]).rad.mean()
def win_dmt(s, d, k):
    v = [dmt[(s, x)] for x in range(int(d) - k, int(d)) if (s, x) in dmt.index]
    return float(np.mean(v)) if v else np.nan
F["rad_out_3d_dmt"] = [win_dmt(s, d, 3) for s, d in zip(R.dmt_seg, R.dmt_day)]

# ---------------- episode-cumulative (D): mean ET over current hi10 episode so far (incl. today) ----------------
eps = []
for r in F.index:
    if R.at[r, "hi10"] == 1:
        ch = [r] + [p for p in back_chain(r, 20)]
        ch = [x for x in ch if R.at[x, "hi10"] == 1 and R.at[x, "ep_id"] == R.at[r, "ep_id"]]
        eps.append(nmean([F.at[x, "et_total"] for x in ch]))
    else:
        eps.append(np.nan)
F["ep_mean_et_sofar"] = eps

F = F.reset_index()
F.to_csv(os.path.join(HERE, "vars_v1.csv"), index=False, encoding="utf-8")
RUN.to_csv(os.path.join(HERE, "vars_running_v1.csv"), index=False, encoding="utf-8")

U, U5, UID, UCAL, D = "U", "U5", "Uid", "Ucal(desc)", "D"
desc = {
    "rad_out_sum": ("실외 일사 하루 합 MJ/m2 (바닥 5 W/m2 제거)", U),
    "rad_in_sum": ("실내 일사 대리 합 (투과 .65 · 차광 60% · 보온 40% 차단 가정)", U),
    "shade_loss": ("차광·보온으로 잃은 일사 비율", U),
    "vpd_day": ("낮 09~16시 평균 VPD kPa", U),
    "vpd_24": ("24시간 평균 VPD", U),
    "vpd_rad": ("Σ VPD × radIn (증산 구동 대리)", U),
    "et_day": ("ET 대리(0.30·radIn + 20·VPD) 낮 합 mm", U),
    "et_night": ("ET 대리 밤(일사 ≤10) 10·VPD 합 mm", U),
    "et_total": ("ET 대리 하루 합 mm", U),
    "et_night_share": ("밤 ET / 하루 ET", U),
    "vpd_night": ("밤 평균 VPD", U),
    "hum_night": ("밤 평균 실내 습도", U),
    "cond_hours": ("결로 위험 시간 수 (Tin - 이슬점 < 1℃)", U),
    "dew_margin_min": ("하루 최소 (Tin - 이슬점)", U),
    "vent_vpd": ("Σ 환기/100 × VPD (환기 건조)", U),
    "vent_ahx": ("Σ 환기/100 × max(AH실내-AH실외,0) (환기로 내보낸 수증기 대리)", U),
    "sealed_hours": ("환기 < 1 시간 수 (24h)", U),
    "sealed_day_hours": ("낮 09~16시 중 환기 < 1 시간 수", U),
    "heat_deg_h": ("Σ max(Tin - Tout, 0) (열 요구 대리)", U),
    "heating_sum": ("Σ 난방/100", U),
    "sunrise_hour": ("일사 ≥ 50 첫 시각", U),
    "morning_ramp": ("08~10시 평균 일사", U),
    "ah_rise_morning": ("07~13시 최대 AH - 06시 AH (아침 증산 시작 대리)", U),
    "hum_drop_morning": ("06시 습도 - 08~13시 최저 습도", U),
    "dah_closed_day": ("낮 밀폐·무포그 시간의 평균 시간당 AH 증가 (PP1 재방문)", U),
    "ahx_day": ("낮 AH실내 - AH실외 평균 (PP1 재방문)", U),
    "tdiff_day": ("낮 Tin - Tout 평균", U),
    "vpd_0_5": ("0~5시 평균 VPD", U5), "hum_0_5": ("0~5시 평균 습도", U5), "cond_0_5": ("0~5시 결로 위험 시간", U5),
    "sealed_0_5": ("0~5시 밀폐 시간", U5), "heat_deg_0_5": ("0~5시 열 요구", U5),
    "prev_et_total": ("clean 전날(truth a1 & DM11 gap1) ET 하루 합", D),
    "prev_rad_in_sum": ("clean 전날 실내 일사 대리 합", D),
    "prev_sealed_hours": ("clean 전날 밀폐 시간", D),
    "d_et_total": ("ET 오늘 - clean 전날", D),
    "et_3link": ("clean 링크로 거슬러 간 최대 3일 ET 평균", D),
    "sealed_3link": ("clean 링크 최대 3일 밀폐 시간 평균", D),
    "vpd_night_3link": ("clean 링크 최대 3일 밤 VPD 평균", D),
    "lag2id_et_total": ("같은 ID num-2 학습 기록의 ET 하루 합 (clean 전날과 일치 48%: 1차 53%, 2차 17%)", UID),
    "lag2id_sealed_hours": ("같은 ID num-2 학습 기록의 밀폐 시간", UID),
    "rad_out_3d": ("DM11 달력 d-1..d-3 학습 기록 실외 일사 합 평균", UCAL),
    "rad_out_7d": ("DM11 달력 d-1..d-7 평균", UCAL),
    "cloud_idx": ("오늘 일사 / DM11 ±7일 학습 기록 일사 최대 (계절 대비 흐림)", UCAL),
    "dull_7d": ("지난 7일 중 cloud_idx < .5 날 수", UCAL),
    "rad_out_3d_dmt": ("DMT3e(학습 전용) 같은 구간 d-1..d-3 실외 일사 평균", "Ucal"),
    "ep_mean_et_sofar": ("현재 hi10 에피소드 오늘까지 ET 평균", D),
}
CD = pd.DataFrame([(c, *desc[c]) for c in F.columns if c != "rec"], columns=["variable", "definition", "usability"])
CD.to_csv(os.path.join(HERE, "vars_v1_columns.csv"), index=False, encoding="utf-8-sig")
print("vars", len(CD), "records", len(F))
print(F.drop(columns="rec").describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())
print("running rows", len(RUN))
