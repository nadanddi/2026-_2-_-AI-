# -*- coding: utf-8 -*-
"""재분석 1단계 (2026-10-01, 집 클로드): 데이터 품질 진단 (data-profiling).

기존 카탈로그를 전제로 두지 않고 원자료를 처음부터 진단한다.
정제는 하지 않고 진단만 출력한다. 결과: local/re01_profile.txt
"""
import env  # noqa: F401
import os
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

OUT = os.path.join(env.LOCAL, "re01_profile.txt")
os.makedirs(env.LOCAL, exist_ok=True)
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _f.write(s + "\n")


def section(t):
    p("\n" + "=" * 100)
    p(t)
    p("=" * 100)


# ---------------------------------------------------------------- 원본 로드
X_raw = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
y_raw = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
T_raw = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
S_raw = pd.read_csv(os.path.join(env.DATA, "sample_submission.csv"))


def keys(df):
    k = df["row_id"].str.extract(r"^(?P<farm>[^_]+)_(?P<day>\d+)_(?P<hour>\d+)$")
    k["day"] = k["day"].astype(int)
    k["hour"] = k["hour"].astype(int)
    return k


section("1. 형태·타입·메모리")
for name, df in [("train_X", X_raw), ("train_y", y_raw), ("test_X", T_raw), ("sample_sub", S_raw)]:
    p(f"{name}: {df.shape}, 메모리 {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    p("  dtypes:", dict(df.dtypes.astype(str).value_counts()))

section("2. 키(row_id) 무결성")
for name, df in [("train_X", X_raw), ("train_y", y_raw), ("test_X", T_raw), ("sample_sub", S_raw)]:
    k = keys(df)
    bad = k["farm"].isna().sum()
    p(f"{name}: 형식 불량 {bad}, row_id 중복 {df['row_id'].duplicated().sum()}, "
      f"시각 범위 {k['hour'].min()}~{k['hour'].max()}, 온실 수 {k['farm'].nunique()}")

kx, ky, kt = keys(X_raw), keys(y_raw), keys(T_raw)
X = pd.concat([kx, X_raw], axis=1)
Y = pd.concat([ky, y_raw], axis=1)
T = pd.concat([kt, T_raw], axis=1)

p("\ntrain_y 온실별 행 수:", dict(Y["farm"].value_counts()))
p("test_X 온실별 행 수:", dict(T["farm"].value_counts()))
p("sample_sub == test_X row_id 순서 일치:", bool((S_raw["row_id"].values == T_raw["row_id"].values).all()))
jx = set(X_raw["row_id"])
p("train_y row_id 중 train_X에 있는 비율:", np.mean([r in jx for r in y_raw["row_id"]]))
p("test_X row_id 중 train_X와 겹치는 수:", sum(r in jx for r in T_raw["row_id"]))

section("3. 온실×날 커버리지 (하루 24행인지, 빈 날)")
g = X.groupby(["farm", "day"]).size()
p("train_X 날별 행 수 분포:", dict(g.value_counts().sort_index()))
cov = X.groupby("farm")["day"].agg(["min", "max", "nunique"])
cov["span"] = cov["max"] - cov["min"] + 1
cov["빈날"] = cov["span"] - cov["nunique"]
p("온실별 일차 범위 (앞 10 + F13/F47):")
p(cov.head(10).to_string())
p(cov.loc[["F13", "F47"]].to_string())
p("빈 날이 있는 온실 수:", int((cov["빈날"] > 0).sum()), "/", len(cov))
for f in ["F13", "F47"]:
    dx = sorted(X.loc[X.farm == f, "day"].unique())
    dy = sorted(Y.loc[Y.farm == f, "day"].unique())
    dt = sorted(T.loc[T.farm == f, "day"].unique())
    p(f"{f}: train_X 일 {dx[0]}~{dx[-1]} ({len(dx)}일), train_y 일 {dy[0]}~{dy[-1]} ({len(dy)}일), "
      f"test 일 {dt[0]}~{dt[-1]} ({len(dt)}일)")
    p(f"   train_X에 있지만 y 없는 날: {sorted(set(dx) - set(dy))[:40]}")
    miss = sorted(set(range(dx[0], dt[-1] + 1)) - set(dx) - set(dt))
    p(f"   {dx[0]}~{dt[-1]} 중 어디에도 없는 날: {miss}")
    p(f"   test 날: {dt}")
    p(f"   test 날 중 train_X에도 있는 날: {sorted(set(dt) & set(dx))}")
    gy = Y[Y.farm == f].groupby("day").size()
    p(f"   train_y 날별 행 수 분포: {dict(gy.value_counts().sort_index())}")

section("4. 컬럼별 결측 (train 전체 / train F13·F47 / test)")
cols = [c for c in X_raw.columns if c != "row_id"]
lab = X[X.farm.isin(["F13", "F47"])]
m = pd.DataFrame({
    "train전체%": X[cols].isna().mean() * 100,
    "trainF13F47%": lab[cols].isna().mean() * 100,
    "test%": T[cols].isna().mean() * 100,
})
p(m.round(2).to_string())
p("\ntrain_y 결측:", dict(y_raw.isna().sum()))

section("5. 수치 요약 (train F13·F47 / test / 다른 온실)")
other = X[~X.farm.isin(["F13", "F47"])]
for c in cols:
    a, b, o = lab[c].dropna(), T[c].dropna(), other[c].dropna()
    if len(a) == 0 and len(b) == 0:
        p(f"{c:12s} 전부 결측(F13F47·test)")
        continue
    def q(s):
        if len(s) == 0:
            return "—"
        return (f"min {s.min():8.2f} p1 {s.quantile(.01):8.2f} p50 {s.median():8.2f} "
                f"p99 {s.quantile(.99):8.2f} max {s.max():8.2f} 0비율 {(s==0).mean():.2f} 고유 {s.nunique()}")
    p(f"{c:12s} F13F47 | {q(a)}")
    p(f"{'':12s} test   | {q(b)}")
    p(f"{'':12s} 타온실 | {q(o)}")
p("\ntrain_y:")
for c in ["sub_temp", "sub_ec"]:
    for f in ["F13", "F47"]:
        s = Y.loc[Y.farm == f, c].dropna()
        p(f"  {c} {f}: n {len(s)} min {s.min():.3f} p1 {s.quantile(.01):.3f} p50 {s.median():.3f} "
          f"p99 {s.quantile(.99):.3f} max {s.max():.3f} 음수 {(s<0).sum()} 고유 {s.nunique()}")

section("6. 물리적으로 불가능/의심 값")
chk = {
    "in_hum>100": lambda d: d["in_hum"] > 100,
    "out_hum>100": lambda d: d["out_hum"] > 100,
    "hum<0": lambda d: (d["in_hum"] < 0) | (d["out_hum"] < 0),
    "act_*<0 or >100": lambda d: ((d[[c for c in cols if c.startswith("act_")]] < 0) |
                                  (d[[c for c in cols if c.startswith("act_")]] > 100)).any(axis=1),
    "in_co2<300": lambda d: d["in_co2"] < 300,
    "in_co2>3000": lambda d: d["in_co2"] > 3000,
    "out_rad<0": lambda d: d["out_rad"] < 0,
    "out_wspd<0": lambda d: d["out_wspd"] < 0,
    "in_temp<0 or >45": lambda d: (d["in_temp"] < 0) | (d["in_temp"] > 45),
}
for k_, fn in chk.items():
    p(f"{k_:20s} F13F47 {int(fn(lab).sum()):6d}  test {int(fn(T).sum()):5d}  타온실 {int(fn(other).sum()):7d}")
yl = Y.dropna(subset=["sub_temp"])
p("sub_temp<0 또는 >40:", int(((yl.sub_temp < 0) | (yl.sub_temp > 40)).sum()))

section("7. 중복: 완전 중복 행 / 입력 벡터 중복")
p("train_X 입력 완전 중복 행(row_id 제외):", int(X_raw[cols].duplicated().sum()))
p("F13F47 입력 완전 중복 행:", int(lab[cols].duplicated().sum()))
p("test 입력 완전 중복 행:", int(T_raw[cols].duplicated().sum()))

section("8. 고착(같은 값 연속) — 실내 3종, F13·F47")
def runs(s):
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).transform("size")
for f in ["F13", "F47"]:
    d = X[X.farm == f].sort_values(["day", "hour"])
    for c in ["in_temp", "in_hum", "in_co2"]:
        r = runs(d[c])
        p(f"  {f} {c}: 6시간 이상 같은 값 행 {int((r>=6).sum())}, 12시간 이상 {int((r>=12).sum())}")
    dt_ = T[T.farm == f].sort_values(["day", "hour"])
    for c in ["in_temp", "in_hum", "in_co2"]:
        r = runs(dt_[c])
        p(f"  {f} test {c}: 6시간 이상 같은 값 행 {int((r>=6).sum())}")
yy = Y.sort_values(["farm", "day", "hour"])
for f in ["F13", "F47"]:
    r = runs(yy.loc[yy.farm == f, "sub_temp"])
    p(f"  {f} sub_temp 6시간 이상 같은 값 행: {int((r>=6).sum())}")

section("9. 결측이 무작위인가 — F13·F47 결측 행의 날·시각 분포와 정답 관계")
L = lab.merge(Y[["row_id", "sub_temp"]], on="row_id", how="left")
for c in ["in_temp", "in_hum", "in_co2", "out_temp"]:
    mm = L[c].isna()
    if mm.sum() == 0:
        continue
    p(f"{c}: 결측 {int(mm.sum())}행, 날 {L.loc[mm,'day'].nunique()}개 "
      f"(상위: {dict(L.loc[mm,'day'].value_counts().head(6))}), "
      f"sub_temp 평균 결측행 {L.loc[mm,'sub_temp'].mean():.2f} vs 나머지 {L.loc[~mm,'sub_temp'].mean():.2f}")
mt = L["sub_temp"].isna()
p(f"sub_temp 결측(라벨 없음) {int(mt.sum())}행, 날: {sorted(L.loc[mt,'day'].unique())[:60]}")

section("10. 시간 연속성 — 자정과 다른 시각의 변화량 (F13·F47, 다른 온실 비교)")
def jump_ratio(d, c):
    d = d.sort_values(["day", "hour"])
    dd = d[c].diff()
    contig = d["day"].diff().fillna(0).isin([0, 1])
    mid = (d["hour"] == 0) & contig
    oth = (d["hour"] != 0)
    return dd[mid].abs().mean() / dd[oth].abs().mean()
rows = []
for f, d in X.groupby("farm"):
    rows.append((f, jump_ratio(d, "in_temp"), jump_ratio(d, "in_co2"), jump_ratio(d, "out_temp")))
jr = pd.DataFrame(rows, columns=["farm", "in_temp", "in_co2", "out_temp"]).set_index("farm")
p("자정 변화/다른 시각 변화 비 — F13·F47:")
p(jr.loc[["F13", "F47"]].round(2).to_string())
p("다른 온실 분포(in_temp): ", jr.drop(["F13", "F47"])["in_temp"].describe().round(2).to_dict())
for f in ["F13", "F47"]:
    d = Y[Y.farm == f]
    p(f"sub_temp {f} 자정 비: {jump_ratio(d, 'sub_temp'):.2f}")
    p(f"test in_temp {f} 자정 비: {jump_ratio(T[T.farm==f], 'in_temp'):.2f}")

_f.close()
print("\n저장:", OUT)
