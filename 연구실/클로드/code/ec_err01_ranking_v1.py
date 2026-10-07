# -*- coding: utf-8 -*-
"""EC 오차 순위 정리 (진단 전용, 채택 판정 아님)  — 2026-10-06 연구실 클로드

기준 모델: 저장 OOF `집/클로드/research/local/ec3_DP1_all.csv` 의 dp_{7,101,2024}
  = 계절 R3(R3S) + DP1 운영 순서 특징, 시드 평균. (TabPFN 0.2 혼합·SG2 후처리는 빠짐)
비교: 같은 파일 r3s (DP1 없음).
검증기: DIAG10 (공개 정답 360일 전부를 10폴드로 한 번씩 검증) 이 주,
        A/B 는 같은 순위가 유지되는지 보조 확인.

산출: 제곱오차(SSE) 몫으로 문제 순위.
  1) 하루 수준(일평균 잔차) vs 하루 안 모양
  2) 서로 겹치지 않는 날 유형별 SSE 몫
  3) 겹치는 요인(2차 구간, 쌍 역할, 밀폐, 온실, 시각, 추위)별 몫과 행 비율
  4) 최악 날 목록
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "..", "results")
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")

P = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv"))
P["pred"] = P[["dp_7", "dp_101", "dp_2024"]].mean(axis=1)
P["pred_r3s"] = P[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X["farm"] = X.row_id.str[:3]; X["day"] = X.row_id.str[4:7].astype(int)
roles = pd.read_csv(os.path.join(H, "st_dong_assign_v1.csv"))

# 날 단위 입력 요약 (학습 입력만, 진단용)
g = X.groupby(["farm", "day"])
DX = pd.DataFrame({
    "in_temp": g.in_temp.mean(), "out_temp": g.out_temp.mean(),
    "sealed": (g.act_circfan.mean() < 10) & (g.act_vent.apply(lambda s: (s == 0).mean()) > 0.85),
}).reset_index()


def analyse(V, pcol, tag, verbose=True):
    D = P[P.validator == V].copy()
    D["res"] = D[pcol] - D.sub_ec
    # 같은 행이 여러 폴드에 나오는 검증기(A/B)는 행 단위로 평균하지 않고 그대로 출현 단위로 센다
    D["key"] = D.farm + "_" + D.day.astype(str) + "_" + D.validation_fold.astype(str)
    day = D.groupby("key").agg(farm=("farm", "first"), day=("day", "first"), y=("sub_ec", "mean"),
                               p=(pcol, "mean"), bias=("res", "mean"), n=("res", "size"),
                               sse=("res", lambda r: (r ** 2).sum())).reset_index()
    day["sse_level"] = day.n * day.bias ** 2
    day["sse_shape"] = day.sse - day.sse_level
    day = day.merge(roles[["farm", "day", "role", "dong"]], on=["farm", "day"], how="left").merge(DX, on=["farm", "day"], how="left")
    day["late"] = day.day >= 179
    tq = day.in_temp.quantile(0.2)
    day["cold"] = day.in_temp <= tq
    T = day.sse.sum(); N = day.n.sum()
    lines = []
    pr = lines.append
    pr("=== %s / %s : RMSE %.4f, 날 %d, 행 %d" % (V, pcol, np.sqrt(T / N), len(day), N))
    pr("[1] 하루 수준 몫 %.1f%% / 하루 안 모양 몫 %.1f%%" % (100 * day.sse_level.sum() / T, 100 * day.sse_shape.sum() / T))

    # [2] 서로 배타적인 날 유형
    def typ(r):
        if r.y >= 1.0:
            return "H1 고EC 날(일평균≥1) 과소" if r.bias < 0 else "H2 고EC 날 과대"
        if r.p >= 0.9:
            return "N1 일반 날 오탐(예측≥.9)"
        if r.bias > 0.1:
            return "N2 일반 날 과대(+.1 초과)"
        if r.bias < -0.1:
            return "N3 일반 날 과소(−.1 미만)"
        return "N4 일반 날 수준 맞음(|편향|≤.1)"
    day["type"] = day.apply(typ, axis=1)
    t = day.groupby("type").agg(days=("sse", "size"), sse=("sse", "sum"), lvl=("sse_level", "sum"),
                                y=("y", "mean"), p=("p", "mean"), bias=("bias", "mean"))
    t["몫%"] = 100 * t.sse / T; t["그중수준%"] = 100 * t.lvl / t.sse; t["날%"] = 100 * t.days / len(day)
    pr("[2] 날 유형 (서로 배타)\n" + t.sort_values("sse", ascending=False)[["days", "날%", "몫%", "그중수준%", "y", "p", "bias"]].round(3).to_string())

    # [3] 겹치는 요인
    rows = []
    def fac(name, m):
        rows.append((name, m.sum(), 100 * day.n[m].sum() / N, 100 * day.sse[m].sum() / T,
                     np.sqrt(day.sse[m].sum() / day.n[m].sum()) if m.any() else np.nan))
    fac("2차 구간(day≥179)", day.late)
    fac("1차 구간", ~day.late)
    for f in ("F13", "F47"):
        fac("온실 " + f, day.farm == f)
    for r_ in ("single", "first", "second"):
        fac("기록 역할 " + r_, day.role == r_)
        fac("  2차 & " + r_, (day.role == r_) & day.late)
    fac("고EC 동(B)", day.dong == "B")
    fac("밀폐 날", day.sealed == True)
    fac("추운 날(실내 하위20%)", day.cold)
    fac("고EC 날 31류(y≥1)", day.y >= 1)
    fac("2차 & 고EC", day.late & (day.y >= 1))
    fac("2차 & 일반", day.late & (day.y < 1))
    f = pd.DataFrame(rows, columns=["요인", "날", "행%", "SSE몫%", "RMSE"])
    f["배율(몫/행)"] = f["SSE몫%"] / f["행%"]
    pr("[3] 겹치는 요인\n" + f.round(3).to_string(index=False))

    # 시각별
    hh = D.groupby("hour").res.apply(lambda r: (r ** 2).sum())
    hb = D.groupby("hour").res.mean()
    pr("[3b] 시각별 SSE 몫%% (상위6): " + ", ".join("%dh %.1f%%(편향%+.3f)" % (h, 100 * hh[h] / T, hb[h]) for h in hh.sort_values(ascending=False).index[:6]))
    pr("     시각별 SSE 몫%% (하위3): " + ", ".join("%dh %.1f%%" % (h, 100 * hh[h] / T) for h in hh.sort_values().index[:3]))

    # [4] 집중도·최악 날
    s = day.sse.sort_values(ascending=False).values
    pr("[4] 집중도: 상위 5날 %.1f%%, 10날 %.1f%%, 20날 %.1f%%, 36날(10%%) %.1f%%" % tuple(100 * s[:k].sum() / T for k in (5, 10, 20, 36)))
    w = day.sort_values("sse", ascending=False).head(25)
    w = w.assign(몫=100 * w.sse / T)
    pr(w[["farm", "day", "role", "dong", "late", "sealed", "y", "p", "bias", "몫", "in_temp", "type"]].round(3).to_string(index=False))
    text = "\n".join(lines)
    if verbose:
        print(text)
    return day, text


if __name__ == "__main__":
    texts = []
    day, t = analyse("DIAG10", "pred", "dp"); texts.append(t)
    day.to_csv(os.path.join(OUT, "ec_err01_days_DIAG10_v1.csv"), index=False, encoding="utf-8-sig")
    # 시드별 안정성: 날 유형 몫
    print("\n=== 시드별 날 유형 몫(DIAG10) ===")
    for s in (7, 101, 2024):
        _, t = analyse("DIAG10", "dp_%d" % s, "s", verbose=False)
        blk = t.split("[2]")[1].split("[3]")[0]
        print("seed", s, blk); texts.append("seed %d [2]%s" % (s, blk))
    for V in ("A", "B"):
        _, t = analyse(V, "pred", "dp", verbose=False)
        print(t.split("[3b]")[0]); texts.append(t)
    _, t = analyse("DIAG10", "pred_r3s", "r3s", verbose=False)
    print(t.split("[3]")[0]); texts.append(t)
