# -*- coding: utf-8 -*-
"""RZ5 v2 (분할 버그 수정: 짝수 반올림 → floor, 기준 동일) (2026-10-12 집 클로드; 사용자: "EC가 올라가는데 필요한 조건들이 있는 거 아닐까? 양액을 최근에 줬는데 금방 또 줄 수도 없을 것 같고,
일정 온도가 반복되거나 일정 습도가 지속되거나…").  서술 포렌식(학습 라벨), 실행 전 고정:
사건: 하루 안 상승 up = EC25(20..23) − EC25(0..3) >= .10 (주), >= .15 (보조).
후보 조건(모두 그날 0..12시 또는 이전 날짜; 대상 기록 이후 정보 없음):
  그날 0..12시: 실내온 ≥t 시간 수, 습도 ≥t 시간 수, 최저 습도, 최대 실내온, VPD 평균, 일사 합, 천창>0 시간, 난방>0 시간, 차광 평균, 유동팬 평균, 실내−외기 차 평균
  이전 날짜(같은 농장 달력 −1·−2·−3일, 1차만): 외기 평균·최고·최저, 일사 합, 실내 평균(같은 농장 기록 중앙), 습도 평균, 3일 합/평균
  상태: 0시 EC25 수준, 달력 위치(1차)
판정: 각 조건×방향×문턱(10..90% 분위수)에서 coverage = 사건 중 조건 충족 비율, base = 전체 중 충족 비율.
  '필요 조건' 후보 = coverage >= .90 이고 base <= .50 (선택: 1차 홀수 달력 날짜). 확인 = 1차 짝수 날짜와 2차 각각에서
  coverage >= .80, base <= .60, Fisher 단측 p < .05 (둘 다).  고EC 상태가 섞인 효과를 가리려고 0시 EC25 >= .9 기록만으로도 반복.
쉬는 기간(사용자 ①): 확실 연결 사슬(같은 동 연속일)에서 상승 다음 날 상승 확률 vs 기준.
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from scipy.stats import fisher_exact
TRi = np.where(TRAIN)[0]
A = pd.read_csv("rz2_records_v1.csv", index_col=0)
T_, H_, C_ = IN_[:, :, 0], IN_[:, :, 1], IN_[:, :, 2]
vent, shade, thermal, heat, fan = [AC[:, :, k] for k in range(5)]
OT, RAD = X[:, :, 0], X[:, :, 2]
es = lambda t: .6108 * np.exp(17.27 * t / (t + 237.3))
VPD = es(T_) * (1 - H_ / 100)
M = slice(0, 13)
F = pd.DataFrame(index=TRi)
for t in (14, 16, 18, 20, 22, 25): F[f"hrs_tin>={t}"] = (T_[TRi, M] >= t).sum(1)
for t in (70, 80, 85, 90): F[f"hrs_hum>={t}"] = (H_[TRi, M] >= t).sum(1)
F["min_hum"] = np.nanmin(H_[TRi, M], 1); F["max_tin"] = np.nanmax(T_[TRi, M], 1); F["vpd"] = np.nanmean(VPD[TRi, M], 1)
F["rad_0_12"] = np.nansum(RAD[TRi, M], 1); F["hrs_vent"] = (vent[TRi, M] > 0).sum(1); F["hrs_heat"] = (heat[TRi, M] > 0).sum(1)
F["shade"] = np.nanmean(shade[TRi, M], 1); F["fan"] = np.nanmean(fan[TRi, M], 1); F["in_out"] = np.nanmean(T_[TRi, M] - OT[TRi, M], 1)
F["co2"] = np.nanmean(C_[TRi, M], 1)
F["lvl0"] = A.loc[TRi, "lvl0"].values
# previous calendar dates (pass 1)
P1 = A[~A.p2 & A.cal.notna()]; dkey = np.round(P1.cal).astype(int)
daily = pd.DataFrame(dict(farm=P1.farm, d=dkey, ot=np.nanmean(OT[P1.index], 1), otmax=np.nanmax(OT[P1.index], 1), otmin=np.nanmin(OT[P1.index], 1),
                          rad=np.nansum(RAD[P1.index], 1), tin=np.nanmean(T_[P1.index], 1), hum=np.nanmean(H_[P1.index], 1), vpdd=np.nanmean(VPD[P1.index], 1))).groupby(["farm", "d"]).median()
for v in ("ot", "otmax", "otmin", "rad", "tin", "hum", "vpdd"):
    for k in (1, 2):
        F[f"{v}_d-{k}"] = [daily[v].get((A.at[i, "farm"], int(round(A.at[i, "cal"])) - k), np.nan) if (not A.at[i, "p2"] and np.isfinite(A.at[i, "cal"])) else np.nan for i in TRi]
    F[f"{v}_3d"] = [np.nanmean([daily[v].get((A.at[i, "farm"], int(round(A.at[i, "cal"])) - k), np.nan) for k in (1, 2, 3)]) if (not A.at[i, "p2"] and np.isfinite(A.at[i, "cal"])) else np.nan for i in TRi]
F["cal"] = A.loc[TRi, "cal"].values
F["p2"] = A.loc[TRi, "p2"].values; F["up"] = A.loc[TRi, "up"].values
F["odd"] = [(int(np.floor(c)) % 2 == 1) if np.isfinite(c) else False for c in F.cal]
feats = [c for c in F.columns if c not in ("p2", "up", "odd")]
def scan(sel_mask, ev, label):
    S = F[sel_mask]; e = ev[sel_mask]; out = []
    for f in feats:
        v = S[f]; ok = v.notna()
        if ok.sum() < 30 or e[ok].sum() < 5: continue
        for q in np.arange(.1, .91, .1):
            t = v[ok].quantile(q)
            for dr in (">=", "<="):
                c = (v >= t) if dr == ">=" else (v <= t)
                cov = c[ok & e].mean(); base = c[ok].mean()
                out.append((f, dr, t, cov, base))
    return pd.DataFrame(out, columns=["f", "dir", "t", "cov", "base"])
def check(mask, ev, f, dr, t):
    S = F[mask]; e = ev[mask]; v = S[f]; ok = v.notna()
    if ok.sum() < 10 or e[ok].sum() < 2: return None
    c = (v >= t) if dr == ">=" else (v <= t)
    tab = [[int((c & e & ok).sum()), int((c & ~e & ok).sum())], [int((~c & e & ok).sum()), int((~c & ~e & ok).sum())]]
    return c[ok & e].mean(), c[ok].mean(), fisher_exact(tab, alternative="greater")[1], int(e[ok].sum())
for TH in (.10, .15):
    ev = F.up >= TH
    for scope, base_mask in (("all records", F.up.notna()), ("0h EC25>=.9 only", F.lvl0 >= .9)):
        print(f"\n===== event up >= {TH:.2f} | {scope} =====")
        selA = base_mask & ~F.p2 & F.odd
        sc = scan(selA, ev, "A"); cand = sc[(sc["cov"] >= .9) & (sc.base <= .5)].copy()
        cand["gap"] = cand["cov"] - cand.base; cand = cand.sort_values("gap", ascending=False).drop_duplicates("f").head(15)
        print("selection (pass-1 odd dates): events %d of %d | candidates %d" % (int((ev & selA).sum()), int(selA.sum()), len(cand)))
        for r in cand.itertuples():
            b = check(base_mask & ~F.p2 & ~F.odd, ev, r.f, r.dir, r.t); c2 = check(base_mask & F.p2, ev, r.f, r.dir, r.t)
            fmt = lambda x: ("cov %.2f base %.2f p %.3f (ev %d)" % x) if x else "n/a"
            ok = all(x is not None and x[0] >= .8 and x[1] <= .6 and x[2] < .05 for x in (b, c2)) if not r.f.endswith(("_d-1", "_d-2", "_3d")) and r.f != "cal" else (b is not None and b[0] >= .8 and b[1] <= .6 and b[2] < .05)
            print(f"  {r.f:14s} {r.dir} {r.t:9.2f} | A cov {r.cov:.2f} base {r.base:.2f} | B even-dates {fmt(b)} | pass-2 {fmt(c2)} | {'PASS' if ok else 'fail'}")
# refractory: certain-link chains
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLa = TLa[TLa.cert & (TLa.fa == TLa.fb)]
nx = {int(a): int(b) for a, b in zip(TLa.a1, TLa.b) if TRAIN[int(a)] and TRAIN[int(b)]}
ev = (F.up >= .10)
pairs = [(bool(ev[a]), bool(ev[b])) for a, b in nx.items() if a in F.index and b in F.index]
pp = np.array(pairs)
print("\n[refractory] same-house next-day pairs %d: P(rise next | rise today) %.2f (n %d) vs P(rise next | no rise) %.2f (n %d)" % (
    len(pp), pp[pp[:, 0], 1].mean() if pp[:, 0].any() else np.nan, pp[:, 0].sum(), pp[~pp[:, 0], 1].mean(), (~pp[:, 0]).sum()))
hi = [(bool(ev[a]), bool(ev[b])) for a, b in nx.items() if a in F.index and b in F.index and F.at[a, "lvl0"] >= .9]
if hi:
    hp = np.array(hi); print("  within high state (prev 0h>=.9): P(rise | rise prev) %.2f (n %d) vs P(rise | no rise prev) %.2f (n %d)" % (
        hp[hp[:, 0], 1].mean() if hp[:, 0].any() else np.nan, hp[:, 0].sum(), hp[~hp[:, 0], 1].mean() if (~hp[:, 0]).any() else np.nan, (~hp[:, 0]).sum()))
F.to_csv("rz5_features_v1.csv")
