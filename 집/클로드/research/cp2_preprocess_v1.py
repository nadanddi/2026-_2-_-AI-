# -*- coding: utf-8 -*-
"""CP2 (2026-10-14 집 클로드; 사용자: '빠뜨린 전처리까지 다 넣어서 정리').  가변 길이 사슬 RNN 전처리 — CP1(6.854) 확장.
[연결]
 1. 날짜: 학습 = DMT3e(seg, day), 평가 = 0시 외기 일치 학습 기록 최빈.  잠금 40일 표시(lock) — 정답은 학습·검증에 안 씀, 입력은 연결용으로 남김.
 2. 1일 연결 = cert(1:1, 비용 낮은 쪽) — 날짜 불일치·하루 운영 지문 거리 > 95% 분위면 끊고 '관리 변화/의심' 표시(CP1 과 같음).
 3. 빈 날 잇기: 사슬 시작마다 같은 구간 날짜 d−2..d−5 의 '끝이 비어 있는' 정답 기록 중 하루 지문 거리 최소, 거리 <= 같은 문턱 & 2등/1등 >= 1.5 이면 잇고 gap_days 기록(1:1).
    -> 동 트랙(track) = 1일 연결 + 빈 날 연결, track_pos·gap_days(앞 기록과 날짜 차; 시작은 0).
 4. 관리 변화(regime) = 끊긴 1일 연결 자리, 또는 트랙 안 연결의 지문 거리 > 95% 분위.
[입력]
 5. 빈 열 제거: in_rad·act_side·act_valve·act_cool·act_pump(F13·F47 전부 결측, 6.545).
 6. 오염 표시: 복원 행(train_flags_v6.restored_flags — V1·V5 등 규칙, 학습 행만), 직선 채움(rd2 규칙: 2차 차분 <= .05 가 in_temp 8시간·in_co2 10시간 이상, 학습 행만),
    거친 CO2 30일(rw1).  수리(학습 행만, 10-10 결정 '학습 입력은 앞뒤로 고쳐도 됨'): 거친 30일의 in_co2·in_hum·in_temp 를 같은 기록 안 중심 이동평균(창 3)으로 — 평가 입력은 그대로.
    flag 열은 RNN 가중치(0.2) 용으로 남김.
 7. 계절: 학습 달력 날짜(seg, day) 를 그대로 입력 후보로(기록 번호 대신 — 현장에선 실제 날짜로 대체 가능).
[목표]
 8. 기본 예측 = 폴드 밖(OOF): ec_rv1 CUR DIAG10 sg 시드 평균(학습 360기록; 잠금은 NaN).  잔차 목표 = log(y) − log(base) (둘 다 .05 아래 자름) 와 y − base 둘 다 저장.
 9. 앞날 정답 가리기 비율 p: 평가 60기록 중 '1일 전 학습 정답 짝이 확신으로 보이는' 비율(CP1: 26/60 ≈ .43)의 여집합 .57 — 이 비율은 평가 입력 구조에서 온 것이라 서술값으로만 기록하고,
    학습 설계 값은 규정상 평가 통계에 맞추지 않도록 .5 로 고정(예외 없이).
[검증]
 10. cut_tracks(hidden) 로 폴드 경계에서 트랙 자르기(cp2_lib_v1).  11. 상황 표시: prev_vis(1일 전 학습 정답 짝) / gap_vis(빈 날 건너 학습 정답 조상) / none.
[평가 규칙]
 12. 평가 기록 앞 기록 후보표: 날짜 d−1..d−5 같은 구간, 학습 기록 또는 같은 ID 앞 번호 평가 기록만, 하루 지문 거리(서술 — 예측 땐 0..h 시로 다시 계산).
출력: cp2_tracks_v1.csv(기록 단위), local/cp2_rows_v1.parquet 대신 cp2_rows_v1.csv.gz(행 단위), cp2_eval_history_v1.csv.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
import json
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
LOCK = {f"{s['farm']}_{int(s['day']):03d}" for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]}
LAB = TRAIN & np.isfinite(DM)
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = np.array([dict(zip(C.rec, C.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(C.rec, C.seg)).get(rid(i), np.nan) for i in range(N)])
def tdate(b):
    vi = np.where(LAB)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
    if len(m) == 0: return np.nan, np.nan
    d = pd.Series(DAYT[m]).mode().iloc[0]; return d, pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
DAY, SEG = DAYT.copy(), SEGT.copy()
for i in range(N):
    if not TRAIN[i]: DAY[i], SEG[i] = tdate(i)
# ---------- inputs ----------
DROP = ["in_rad", "act_side", "act_valve", "act_cool", "act_pump"]
tX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); sX = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
tX["is_train"] = True; sX["is_train"] = False; A = pd.concat([tX, sX]); A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
A["rec"] = A.row_id.str[:7]; A["hour"] = A.row_id.str[8:].astype(int); A = A.drop(columns=[c for c in DROP if c in A.columns]).sort_values(["rec", "hour"]).reset_index(drop=True)
spec = importlib.util.spec_from_file_location("tf6", os.path.join(env.ROOT, u"제출", u"미제출", u"submission_23_일치앵커TNA8_고EC배타B5_가설검정", u"팜모니_정형데이터_재현패키지_23", u"온도", "code", "train_flags_v6.py"))
tf6 = importlib.util.module_from_spec(spec)
try:
    sys.path.insert(0, os.path.dirname(spec.origin)); spec.loader.exec_module(tf6); RF = tf6.restored_flags()
    rf = set(RF.loc[RF["flag"].astype(bool), "row_id"])   # [수정: DataFrame(row_id, V1, V5b, V4nm, flag) 반환 — 첫 판은 전 행 표시 오류]
except Exception as e:
    print("restored_flags unavailable:", repr(e)[:200]); rf = set()
A["flag_restored"] = A.row_id.isin(rf) & A.is_train
def linfill(s, run):
    d2 = s.diff().diff().abs() <= .05; grp = (~d2).cumsum(); ln = d2.groupby(grp).transform("sum")
    return (d2 & (ln >= run)).values
A["flag_linear"] = False
for r, g in A[A.is_train].groupby("rec"):
    A.loc[g.index, "flag_linear"] = linfill(g.in_temp.reset_index(drop=True), 8) | linfill(g.in_co2.reset_index(drop=True), 10)
RW = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv")); rough = {f"{f}_{int(d):03d}" for f, d in zip(RW.farm, RW.day)}
A["flag_rough_day"] = A.rec.isin(rough) & A.is_train
for c in ("in_co2", "in_hum", "in_temp"):
    sm = A.groupby("rec")[c].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    A[c + "_raw"] = A[c]; A.loc[A.flag_rough_day, c] = sm[A.flag_rough_day]
# ---------- base OOF and targets ----------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
base = O.groupby("row_id").sg.mean(); A["base_oof"] = A.row_id.map(base)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")).set_index("row_id").sub_ec; A["y"] = A.row_id.map(Y)
A.loc[A.rec.isin(LOCK), ["y", "base_oof"]] = np.nan; A["lock"] = A.rec.isin(LOCK)
A["resid"] = A.y - A.base_oof; A["log_resid"] = np.log(A.y.clip(lower=.05)) - np.log(A.base_oof.clip(lower=.05))
# ---------- links / tracks ----------
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
FPA = np.stack([A.pivot(index="rec", columns="hour", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)
fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FPA[a] - FPA[b]), 4)))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert].sort_values("c1")
prev, used = {}, set()
for a, b in zip(TLc.a1.astype(int), TLc.b.astype(int)):
    if b in prev or a in used or not (LAB[a] and LAB[b]): continue
    prev[b] = a; used.add(a)
FPCUT = float(np.quantile([fpd(a, b) for b, a in prev.items()], .95))
bad = {b for b, a in prev.items() if (not np.isnan(DAY[a]) and not np.isnan(DAY[b]) and not (DAY[a] == DAY[b] - 1 and SEG[a] == SEG[b])) or fpd(a, b) > FPCUT}
link = {b: (a, 1) for b, a in prev.items() if b not in bad}; hasnext = {a for a, _ in link.values()}
starts = sorted([i for i in np.where(LAB)[0] if i not in link and not np.isnan(DAY[i])], key=lambda i: DAY[i])
ngap = 0
for b in starts:
    cand = [a for a in np.where(LAB)[0] if a != b and a not in hasnext and SEG[a] == SEG[b] and 2 <= DAY[b] - DAY[a] <= 5]
    if not cand: continue
    d = sorted((fpd(a, b), a) for a in cand)
    if d[0][0] <= FPCUT and (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)):
        a = d[0][1]; link[b] = (a, int(DAY[b] - DAY[a])); hasnext.add(a); ngap += 1
nxt = {a: b for b, (a, _) in link.items()}
track, tpos = {}, {}; tid = 0
for i in np.where(LAB)[0]:
    if i in link: continue
    cur, p = i, 0
    while True:
        track[cur], tpos[cur] = tid, p
        if cur not in nxt: break
        cur, p = nxt[cur], p + 1
    tid += 1
km = KMeans(4, random_state=0, n_init=10).fit(np.nan_to_num(FPA[LAB].reshape(LAB.sum(), -1))); fpt = km.predict(np.nan_to_num(FPA.reshape(N, -1)))
rows = []
for i in range(N):
    r = rid(i); a_g = link.get(i)
    rows.append(dict(rec=r, train=bool(LAB[i]), lock=r in LOCK, pass2=NUM[i] >= 179, day=DAY[i], seg=SEG[i], track=track.get(i, -1), track_pos=tpos.get(i, -1),
                     prev=rid(a_g[0]) if a_g else None, gap_days=(a_g[1] if a_g else 0), link_type=("none" if not a_g else ("day1" if a_g[1] == 1 else "gap")),
                     regime_break=bool(i in bad), link_fp=fpd(a_g[0], i) if a_g else np.nan, fp_type=int(fpt[i]), high=bool(LAB[i] and DM[i] >= 1.0), dm=float(DM[i]) if LAB[i] else np.nan))
T = pd.DataFrame(rows); T["track_len"] = T.groupby("track").rec.transform("size").where(T.track >= 0, 0)
T["situation"] = np.select([T.link_type == "day1", T.link_type == "gap"], ["prev_vis", "gap_vis"], "none")
# ---------- eval history ----------
eh = []
for i in np.where(~TRAIN)[0]:
    if np.isnan(DAY[i]): continue
    for a in range(N):
        if a == i or SEG[a] != SEG[i] or not (1 <= DAY[i] - DAY[a] <= 5): continue
        if LAB[a] and rid(a) not in LOCK: kind = "train"
        elif (not TRAIN[a]) and FARM[a] == FARM[i] and NUM[a] < NUM[i]: kind = "eval_same_id_earlier"
        else: continue
        eh.append(dict(rec=rid(i), cand=rid(a), kind=kind, gap_days=int(DAY[i] - DAY[a]), fp=fpd(a, i)))
E = pd.DataFrame(eh)
T.to_csv(os.path.join(HERE, "cp2_tracks_v1.csv"), index=False); A.to_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"), index=False, compression="gzip"); E.to_csv(os.path.join(HERE, "cp2_eval_history_v1.csv"), index=False)
# ---------- summary ----------
tr = T[T.train]; L = tr.drop_duplicates("track").track_len
print(f"links: day1 {sum(1 for _, g in link.values() if g == 1)} | gap links added {ngap} (gap days {pd.Series([g for _, g in link.values() if g > 1]).value_counts().sort_index().to_dict()}) | regime/suspect breaks {len(bad)}")
print(f"tracks {tr.track.nunique()} | length distribution {L.value_counts().sort_index().to_dict()} | records in tracks >= 3: {int((tr.track_len >= 3).sum())}, >= 5: {int((tr.track_len >= 5).sum())}")
print("situations (training records):", tr.situation.value_counts().to_dict(), "| high-EC records by situation:", tr[tr.high].situation.value_counts().to_dict())
print(f"input flags (train rows): restored {int(A.flag_restored.sum())}, linear-fill {int(A.flag_linear.sum())}, rough-day rows {int(A.flag_rough_day.sum())} (repaired by centred mean)")
print(f"base OOF rows {int(A.base_oof.notna().sum())} | resid RMSE (all {A.resid.pow(2).mean() ** .5:.3f}) | lock records {len(LOCK)}")
print(f"eval history: eval records with any allowed predecessor within 5 days: {E.rec.nunique() if len(E) else 0}/60 | by kind {E.drop_duplicates(['rec', 'kind']).kind.value_counts().to_dict() if len(E) else {}}")
print("label-mask design value p = .5 (fixed; eval-structure share 0.57 recorded only as description)")
