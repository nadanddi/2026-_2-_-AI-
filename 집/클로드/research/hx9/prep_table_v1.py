# -*- coding: utf-8 -*-
"""HX9 공용 분석표 v1 (2026-10-13 집 클로드). 사용자 요청: '고EC 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인'.
이 스크립트는 '표 만들기'만 한다 (가설 판정 없음, 서술).
산출:
  hx9/data/records.csv          F13/F47 학습 기록 400개, 기록당 한 줄
  hx9/data/hourly.csv           학습 기록-시간 9600줄
  hx9/data/records_columns.csv  열 설명 + 평가 행에서 쓸 수 있는지(usable / diagnostic / descriptive)
자료 (읽기만):
  env.DATA train_X / train_y (14개 입력, sub_ec, sub_temp)
  linker_v4/feat4: EC, ST, X(실외), FARM, NUM, TRAIN, S3(truth.pkl)
  truth.pkl a1 = 정답 연속성으로 고른 전날 후보 (b 자신의 정답 사용 → 진단용)
  DM11 달력 ec0h/rise/dm11_calendar_v1.csv (평가 입력 사용 → 서술 전용)
  DMT3e 달력 local/dmt3e_E0_days.csv (학습 기록만)
  연구실/클로드/results/ec_rv1_preds_v1.csv  tag=CUR, validator=DIAG10, 시드 3개 평균 = 현재 모델 OOF
    (Codex 잠금 40일은 DIAG10 OOF 에 없음 → sg NaN, locked40=1)
정의:
  clean 전날 링크: truth a1(b) 이고 DM11 day(b) - day(a1) == 1. 다음날(successor) = 그 역방향(여럿이면 c1 최소).
  episode: hi10(하루 평균 >= 1.0) 기록끼리 clean 링크로 이은 연결 성분. ep12 는 hi12(>= 1.2) 기준 같은 방식.
  같은 날 쌍둥이: 학습 기록 중 0시 실외 4변수(소수 3자리 반올림)가 같은 기록. 검산용으로 하루 실외 칸 70% 이상 일치(feat4 방식)도 함께.
  낮 = 09~16시, 밤 = 20~23시 + 0~5시 (같은 기록 안; fh1 과 같음).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RES, "linker_v4"))
from feat4 import EC, ST, X, FARM, NUM, N, TRAIN, S3  # noqa: E402

OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
IDX = {rid(i): i for i in range(N)}
V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
     "act_heating", "act_circfan", "act_co2", "act_fog"]

# ---------------- raw hourly inputs / labels ----------------
TX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy()
TY = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); TY = TY[TY.row_id.str[:3].isin(["F13", "F47"])].copy()
H = TX[["row_id"] + V].merge(TY, on="row_id", how="left")
H["rec"] = H.row_id.str[:7]; H["hour"] = H.row_id.str[8:].astype(int)
H["farm"] = H.rec.str[:3]; H["num"] = H.rec.str[4:].astype(int)
assert len(H) == 9600 and H.rec.nunique() == 400, (len(H), H.rec.nunique())
assert H.groupby("rec").hour.nunique().eq(24).all()

# cross-check labels against feat4 arrays
H["_i"] = H.rec.map(IDX)
assert np.allclose(H.sub_ec.values, EC[H._i.values, H.hour.values], atol=1e-9), "sub_ec mismatch vs feat4"
assert np.allclose(H.sub_temp.values, ST[H._i.values, H.hour.values], atol=1e-9), "sub_temp mismatch vs feat4"

# ---------------- model OOF (DIAG10 CUR seed mean) ----------------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy()
assert set(O.seed.unique()) == {47, 1414, 6464}
SG = O.groupby("row_id").agg(sg=("sg", "mean"), sg_seed_sd=("sg", "std"), fold=("fold", "first"), n_seed=("seed", "nunique")).reset_index()
assert SG.n_seed.eq(3).all()
H = H.merge(SG[["row_id", "sg", "sg_seed_sd", "fold"]], on="row_id", how="left")
H = H.sort_values(["farm", "num", "hour"]).reset_index(drop=True)

# ---------------- per-record label / prediction summaries ----------------
g = H.groupby("rec")
R = pd.DataFrame(index=sorted(H.rec.unique())); R.index.name = "rec"
R["idx"] = [IDX[r] for r in R.index]
R["farm"] = R.index.str[:3]; R["num"] = R.index.str[4:].astype(int); R["pass2"] = (R.num >= 179).astype(int)
ecp = H.pivot(index="rec", columns="hour", values="sub_ec").loc[R.index]
stp = H.pivot(index="rec", columns="hour", values="sub_temp").loc[R.index]
sgp = H.pivot(index="rec", columns="hour", values="sg").loc[R.index]
R["ec_mean"] = ecp.mean(1); R["ec_h0"] = ecp[0]; R["ec_h23"] = ecp[23]; R["ec_max"] = ecp.max(1); R["ec_min"] = ecp.min(1)
R["ec_amp"] = R.ec_max - R.ec_min
R["has_sg"] = sgp.notna().all(1).astype(int); R["locked40"] = 1 - R.has_sg
R["sg_mean"] = sgp.mean(1, skipna=False); R["sg_h0"] = sgp[0]; R["resid_mean"] = R.ec_mean - R.sg_mean
R["sg_seed_sd_mean"] = g.sg_seed_sd.mean().loc[R.index]
R["fold"] = g.fold.first().loc[R.index]
R["hi10"] = (R.ec_mean >= 1.0).astype(int); R["hi12"] = (R.ec_mean >= 1.2).astype(int)
R["st_mean"] = stp.mean(1); R["st_h0"] = stp[0]; R["st_h23"] = stp[23]; R["st_amp"] = stp.max(1) - stp.min(1)

# ---------------- calendars ----------------
C11 = pd.read_csv(os.path.join(RES, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(C11.rec, C11.day))
CT = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv"))
R["dm11_day"] = [D11.get(r, np.nan) for r in R.index]
R = R.join(CT.set_index("rec").rename(columns={"day": "dmt_day", "seg": "dmt_seg"}))

# ---------------- clean predecessor / successor (truth a1 + DM11 gap 1) ----------------
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
assert TL.b.nunique() == len(TL) == 400
assert all(TRAIN[int(a)] for a in TL.a1), "a1 points to an eval record"
recDM = dict(zip(R.idx, R.ec_mean)); recH23 = dict(zip(R.idx, R.ec_h23)); recH0 = dict(zip(R.idx, R.ec_h0))
A1 = {}; PREV = {}; SUCC = {}
for b, a, c1, cert, sf in zip(TL.b, TL.a1, TL.c1, TL.cert, TL.same_farm):
    b, a = int(b), int(a)
    gap = D11.get(rid(b), np.nan) - D11.get(rid(a), np.nan)
    A1[b] = (a, gap, c1, bool(cert), bool(sf))
    if gap == 1:
        PREV[b] = a; SUCC.setdefault(a, []).append((c1, b))
rows = []
for r in R.index:
    i = IDX[r]; a, gap, c1, cert, sf = A1[i]
    d = dict(rec=r, a1_rec=rid(a), a1_dm11_gap=gap, a1_c1=c1, a1_cert=int(cert))
    if i in PREV:
        d.update(prev_rec=rid(a), prev_same_farm=int(sf), prev_ec_mean=recDM[a], prev_ec_h23=recH23[a])
    else:
        d.update(prev_rec=None, prev_same_farm=np.nan, prev_ec_mean=np.nan, prev_ec_h23=np.nan)
    if i in SUCC:
        s = sorted(SUCC[i])[0][1]
        d.update(succ_rec=rid(s), n_succ=len(SUCC[i]), succ_same_farm=int(FARM[s] == FARM[i]), succ_ec_mean=recDM[s], succ_ec_h0=recH0[s])
    else:
        d.update(succ_rec=None, n_succ=0, succ_same_farm=np.nan, succ_ec_mean=np.nan, succ_ec_h0=np.nan)
    rows.append(d)
L = pd.DataFrame(rows).set_index("rec")
R = R.join(L)
R["prev_hi10"] = np.where(R.prev_ec_mean.notna(), (R.prev_ec_mean >= 1.0).astype(float), np.nan)
R["succ_hi10"] = np.where(R.succ_ec_mean.notna(), (R.succ_ec_mean >= 1.0).astype(float), np.nan)
R["link_descriptive"] = 1   # 링크는 b 자신의 정답 + DM11(평가 입력) 사용 → 서술 전용

# ---------------- episodes (connected components along clean links) ----------------
def episodes(flag_col, prefix):
    hi = set(R.index[R[flag_col] == 1])
    par = {r: r for r in hi}
    def fd(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for b, a in PREV.items():
        rb, ra = rid(b), rid(a)
        if rb in hi and ra in hi:
            par[fd(rb)] = fd(ra)
    comp = {r: fd(r) for r in hi}
    groups = {}
    for r, c in comp.items(): groups.setdefault(c, []).append(r)
    # id ordered by (min dm11 day, rec) for stability
    def key(v):
        ds = [R.at[x, "dm11_day"] for x in v]; ds = [d for d in ds if np.isfinite(d)]
        return (min(ds) if ds else 1e9, min(v))
    order = sorted(groups.values(), key=key)
    eid, pos, n, span, ndl = {}, {}, {}, {}, {}
    for k, v in enumerate(order):
        ds = {x: R.at[x, "dm11_day"] for x in v}
        fin = [d for d in ds.values() if np.isfinite(d)]
        d0 = min(fin) if fin else np.nan
        for x in v:
            eid[x] = k; n[x] = len(v); span[x] = (max(fin) - d0 + 1) if fin else np.nan
            pos[x] = ds[x] - d0 if np.isfinite(ds[x]) else np.nan
    R[prefix + "_id"] = [eid.get(r, np.nan) for r in R.index]
    R[prefix + "_pos"] = [pos.get(r, np.nan) for r in R.index]
    R[prefix + "_n"] = [n.get(r, np.nan) for r in R.index]
    R[prefix + "_span"] = [span.get(r, np.nan) for r in R.index]
    return len(order)
n_ep10 = episodes("hi10", "ep")
n_ep12 = episodes("hi12", "ep12")

# ---------------- same-date twins among training records ----------------
X0 = H[H.hour == 0].set_index("rec").loc[R.index, ["out_temp", "out_hum", "out_rad", "out_wspd"]].round(3)
k0 = X0.apply(lambda s: tuple(s.values), axis=1)
grp = {}
for r, k in k0.items(): grp.setdefault(k, []).append(r)
Xfull = np.stack([X[IDX[r]] for r in R.index])            # [400,24,4] feat4 outdoor
tw = []
for j, r in enumerate(R.index):
    t = [x for x in grp[k0[r]] if x != r]
    eq = (np.abs(Xfull - Xfull[j]) < 1e-6).reshape(len(R), -1).mean(1); eq[j] = 0
    tf = [R.index[m] for m in np.where(eq >= 0.7)[0]]
    ev = [R.at[x, "ec_mean"] for x in t]
    tw.append(dict(rec=r, twin_n=len(t), twin_same_farm_n=sum(x[:3] == r[:3] for x in t),
                   twin_ec_max=max(ev) if ev else np.nan, twin_ec_mean=np.mean(ev) if ev else np.nan,
                   twin_hi10_n=sum(e >= 1.0 for e in ev), twin_hi12_n=sum(e >= 1.2 for e in ev),
                   twin_recs=";".join(t), twinfull_n=len(tf), twin_h0_vs_full_agree=int(set(t) == set(tf))))
R = R.join(pd.DataFrame(tw).set_index("rec"))

# ---------------- day / night input means ----------------
day = H[(H.hour >= 9) & (H.hour <= 16)].groupby("rec")[V].mean().add_suffix("_d")
night = H[(H.hour <= 5) | (H.hour >= 20)].groupby("rec")[V].mean().add_suffix("_n")
R = R.join(day).join(night)
R["n_in_nan"] = H.groupby("rec")[["in_temp", "in_hum", "in_co2"]].apply(lambda d: int(d.isna().sum().sum())).loc[R.index]

R = R.reset_index()
R.to_csv(os.path.join(OUT, "records.csv"), index=False, encoding="utf-8")
HO = H[["rec", "farm", "num", "hour", "sub_ec", "sg", "sub_temp"] + V].rename(columns={})
HO.to_csv(os.path.join(OUT, "hourly.csv"), index=False, encoding="utf-8")

# ---------------- column dictionary with eval-row usability ----------------
U, D, S, T = "usable", "diagnostic", "descriptive", "target"
desc = {
    "rec": ("기록 id FARM_NUM", U), "idx": ("feat4 배열 행 번호", U), "farm": ("온실 id", U), "num": ("기록 번호", U),
    "pass2": ("num >= 179 (2차)", U),
    "ec_mean": ("sub_ec 하루 평균", T), "ec_h0": ("sub_ec 0시", T), "ec_h23": ("sub_ec 23시", T), "ec_max": ("sub_ec 하루 최대", T),
    "ec_min": ("sub_ec 하루 최소", T), "ec_amp": ("sub_ec 최대-최소", T),
    "has_sg": ("DIAG10 OOF 있음", S), "locked40": ("Codex 잠금 40일(OOF 없음)", S),
    "sg_mean": ("현재 모델 DIAG10 시드평균 예측 하루 평균 (24시간 입력 필요)", D), "sg_h0": ("현재 모델 0시 예측 = cm0", U),
    "resid_mean": ("ec_mean - sg_mean", D), "sg_seed_sd_mean": ("시드 간 예측 sd 의 하루 평균", D), "fold": ("DIAG10 폴드", S),
    "hi10": ("ec_mean >= 1.0", T), "hi12": ("ec_mean >= 1.2", T),
    "st_mean": ("sub_temp 하루 평균 (다른 과제 정답)", D), "st_h0": ("sub_temp 0시", D), "st_h23": ("sub_temp 23시", D), "st_amp": ("sub_temp 최대-최소", D),
    "dm11_day": ("DM11 달력 날짜 (평가 입력 사용)", S),
    "dmt_day": ("DMT3e 학습 전용 달력 날짜 (평가 기록 위치는 자기 0..h 실외로 학습 기록과 맞춰 추정해야 함)", U + "-in-principle"),
    "dmt_seg": ("DMT3e 구간", U + "-in-principle"),
    "a1_rec": ("truth a1 (b 자신 정답 연속성으로 선택)", D), "a1_dm11_gap": ("DM11 day(b)-day(a1)", S), "a1_c1": ("a1 연속성 비용", D),
    "a1_cert": ("truth cert", D),
    "prev_rec": ("clean 전날 기록 (a1 & DM11 gap 1)", D), "prev_same_farm": ("전날이 같은 온실 id", D),
    "prev_ec_mean": ("전날 EC 하루 평균 (링크가 인과적으로 잡히면 학습 정답이라 usable; 현재 링크는 진단용)", D),
    "prev_ec_h23": ("전날 EC 23시", D), "prev_hi10": ("전날 ec_mean >= 1.0", D),
    "succ_rec": ("clean 다음날 기록 (여럿이면 c1 최소)", D), "n_succ": ("clean 다음날 후보 수", D), "succ_same_farm": ("다음날이 같은 온실 id", D),
    "succ_ec_mean": ("다음날 EC 하루 평균", D), "succ_ec_h0": ("다음날 EC 0시", D), "succ_hi10": ("다음날 ec_mean >= 1.0", D),
    "link_descriptive": ("링크 열은 서술 전용 표시", S),
    "ep_id": ("hi10 에피소드 id (clean 링크 연결 성분)", D), "ep_pos": ("에피소드 안 위치 = DM11 day - 첫 날", D),
    "ep_n": ("에피소드 기록 수", D), "ep_span": ("에피소드 DM11 날짜 폭", D),
    "ep12_id": ("hi12 에피소드 id", D), "ep12_pos": ("hi12 에피소드 위치", D), "ep12_n": ("hi12 에피소드 기록 수", D), "ep12_span": ("hi12 에피소드 날짜 폭", D),
    "twin_n": ("같은 날 쌍둥이 수 (학습 기록 중 0시 실외 4변수 동일)", U), "twin_same_farm_n": ("그중 같은 온실 id", U),
    "twin_ec_max": ("쌍둥이 EC 하루 평균의 최대 (학습 정답)", U), "twin_ec_mean": ("쌍둥이 EC 하루 평균의 평균", U),
    "twin_hi10_n": ("쌍둥이 중 hi10 수", U), "twin_hi12_n": ("쌍둥이 중 hi12 수", U), "twin_recs": ("쌍둥이 목록", U),
    "twinfull_n": ("하루 실외 칸 70% 이상 일치 쌍둥이 수 (24시간 필요 → 검산용)", D),
    "twin_h0_vs_full_agree": ("0시 쌍둥이 집합 == 하루 일치 집합", D),
    "n_in_nan": ("실내 3변수 결측 칸 수", U),
}
for v in V:
    desc[v + "_d"] = (f"{v} 낮(09~16시) 평균 — 평가 행 h>=16 부터 계산 가능", U + "(h>=16)")
    desc[v + "_n"] = (f"{v} 밤(20~23시+0~5시) 평균 — 0~5시 부분은 h>=5, 전체는 h=23", U + "(h=23)")
CD = pd.DataFrame([(c, *desc.get(c, ("", ""))) for c in R.columns], columns=["column", "meaning", "eval_usability"])
assert (CD.meaning != "").all(), CD[CD.meaning == ""]
CD.to_csv(os.path.join(OUT, "records_columns.csv"), index=False, encoding="utf-8-sig")

# ---------------- summary ----------------
print(f"records {len(R)} (F13 {int((R.farm=='F13').sum())}, F47 {int((R.farm=='F47').sum())}; pass2 {int(R.pass2.sum())}); hourly rows {len(HO)}")
print(f"DIAG10 OOF present {int(R.has_sg.sum())}, locked40 (no OOF) {int(R.locked40.sum())}")
for f in ("hi10", "hi12"):
    s = R[R[f] == 1]
    print(f"{f}: n {len(s)} (pass2 {int(s.pass2.sum())}, locked40 {int(s.locked40.sum())}) | clean predecessor {int(s.prev_rec.notna().sum())} "
          f"(of which predecessor hi10 {int((s.prev_hi10 == 1).sum())}) | clean successor {int(s.succ_rec.notna().sum())} "
          f"(successor hi10 {int((s.succ_hi10 == 1).sum())})")
print(f"episodes (chains): hi10 {n_ep10}, hi12 {n_ep12}")
e = R[R.ep_id.notna()].groupby("ep_id").agg(n=("rec", "size"), span=("ep_span", "first"), mx=("ec_mean", "max"))
print("  hi10 episode size distribution:", e.n.value_counts().sort_index().to_dict())
e12 = R[R.ep12_id.notna()].groupby("ep12_id").size()
print("  hi12 episode size distribution:", e12.value_counts().sort_index().to_dict())
allp = R[R.prev_rec.notna()]
print(f"all records with clean predecessor {len(allp)}; P(hi10 | prev hi10) = {allp[allp.prev_hi10==1].hi10.mean():.2f} "
      f"(n {int((allp.prev_hi10==1).sum())}); P(hi10 | prev not hi10) = {allp[allp.prev_hi10==0].hi10.mean():.3f}")
print(f"clean predecessor same farm share {allp.prev_same_farm.mean():.2f}; DM11 NaN {int(R.dm11_day.isna().sum())}")
print(f"twins: records with >=1 h0 twin {int((R.twin_n>0).sum())}; h0 set == full-day set {int(R.twin_h0_vs_full_agree.sum())}/400; "
      f"twin_n dist {R.twin_n.value_counts().sort_index().to_dict()}")
h = R[R.has_sg == 1]
for f in ("hi10", "hi12"):
    s = h[h[f] == 1]
    print(f"model on {f} (with OOF, n {len(s)}): mean resid {s.resid_mean.mean():+.3f}; normal (<1.0) mean resid {h[h.hi10==0].resid_mean.mean():+.3f}")
print("saved:", os.path.join(OUT, "records.csv"), "|", os.path.join(OUT, "hourly.csv"), "|", os.path.join(OUT, "records_columns.csv"))
print("columns:", list(R.columns))
