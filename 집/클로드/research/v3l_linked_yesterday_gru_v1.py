# -*- coding: utf-8 -*-
"""V3L (2026-10-15 집 클로드; 사용자: 5점 모델 계획 → 'C1 하고 V3-L 진행해봐').  사전 고정(커밋 후 실행).  온도.
V3(2층 GRU 128, 48시간 창, 6.531·6.532, 14회차 위 TV3 ADOPT)는 창의 앞 24시간을 '번호 순서상 앞 기록'(같은 농장 t 순서)에서 가져옴 → 같은 동 전날이
  아닐 수 있음(현장 대응 없는 자료 규칙, 계획 v1 의 X 의심).  창을 '같은 동 전날'로 바꾼 판을 만든다.
창 만들기(행 = 기록 r 의 h 시): [전전날 h+1..23] + [전날 0..23] + [r 0..h] 중 마지막 48시간; 전날·전전날은 아래 연결표로 따라감, 없으면 그 부분 NaN(→ 0 + 결측 표시).
  V3L = 연결표 = 링커 v15R 거울(local/linker_v10/v15r15_mirror.csv) prev0, p0 ≥ .5 (대회에서 쓸 수 있는 복원; 사슬 보정 CT2 와 같은 문턱)
  V3T = 연결표 = 진짜 전날 CLEAN(정답 연속 truth.pkl ∩ 달력 차 1) — 현장 조건(C1, 어제를 앎) 서술용
  V30 = 연결 없음(오늘 0..h 만) — 기준 서술용
  입력·학습 = TS7 V3 와 같음(USE 14 + 시각 sin·cos, 표준화 학습 행, Adam 2e-3, 25 에폭, 배치 256, 가중 MSE wb, 시드 1·2 평균).
배치: TS7 의 F·G·H·I(같은 폴드 함수), TS7b 의 DIAG10·EL1·EXT10·EXT12(W.worlds sets).  K 는 저장 V3 가 판 안에 있어 제외(보고에 명시).
판정(TV3 와 같은 틀, v23 사슬·앵커 위, 단일 안 α .025 — 대상은 V3L 만):
  V3L 섞기(b = a + .1 g h (V3L − a), 날 문턱 그대로)가 v23 기본(V3 없음)보다 DIAG10·EL1·F·G·H·I 모든 셀 개선 & 합친(DIAG10+F+G+H+I, 셀 1) 5일 블록 P(악화) < .025
  & EXT10·EXT12 안전(셀 평균 악화이면서 '나음' 비율 < .025 이면 FAIL)  → V3L 이 V3 자리를 맡을 자격.
서술: V3(현행) · V3T · V30 의 같은 지표, V3L vs V3 직접 비교(셀별 %, 합친 P).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u v3l_linked_yesterday_gru_v1.py {cv|sum}
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CK = os.path.join(env.LOCAL, "v3l_ckpt"); KINDS = ("V3L", "V3T", "V30")
def link_maps():
    sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import FARM, NUM, N, S3  # noqa: E402
    rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
    PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv"))
    L = {r: p for r, p, q in zip(PM.rec, PM.prev0, PM.p0) if isinstance(p, str) and q >= .5}
    TL = pd.read_pickle(S3 + r"\truth.pkl"); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
    C = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
    return {"V3L": L, "V3T": C, "V30": {}}
def sequences(lab, a, USE, link):
    a = a.copy(); a["rec"] = a.row_id.str[:7]; a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    cols = USE + ["hs", "hc"]; DAY = {}
    for r, g in a.groupby("rec"):
        m = np.full((24, len(cols)), np.nan, np.float32); g = g.sort_values("hour"); m[g.hour.values.astype(int)] = g[cols].to_numpy(np.float32); DAY[r] = m
    nanday = np.full((24, len(cols)), np.nan, np.float32)
    S = np.full((len(lab), 48, len(cols)), np.nan, np.float32)
    for j, (r, h) in enumerate(zip(lab.row_id.str[:7], lab.hour.astype(int))):
        p1 = link.get(r); p2 = link.get(p1) if p1 else None
        seq = np.concatenate([DAY.get(p2, nanday) if p2 else nanday, DAY.get(p1, nanday) if p1 else nanday, DAY[r][:h + 1]])
        S[j] = seq[-48:]
    return S
def run():
    S7 = load("s7", "ts7_sequence_variants_v1.py"); S6 = S7.S6; W = S6.W
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    tX, ty, sX = W.TM.masked_loader(); a = pd.concat([tX, sX]); a = a[a.farm.isin(["F13", "F47"])]
    LM = link_maps(); SS = {k: sequences(lab, a, S6.USE, LM[k]) for k in KINDS}
    for k in KINDS: print(k, "links", len(LM[k]), "| rows with full 48h window", float(np.isfinite(SS[k][:, 0, 4]).mean()), flush=True)
    jobs = [(L, ff(lab), mk) for L, (ff, mk, *_) in S7.LAY.items()] + [(name, folds, None) for name, folds in sets]
    import common
    for L, folds, mk in jobs:
        for kf, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (L, kf))
            if os.path.exists(path): continue
            tm, vm = (mk(lab, fd) if mk is not None else common.split_mask(lab, fd))
            if not vm.sum(): continue
            out = pd.DataFrame({"row_id": lab.row_id[vm].values})
            for k in KINDS: out[k] = np.mean([S7.fit_predict(SS[k][tm], y[tm], wb[tm], SS[k][vm], s, dev, "V3") for s in (1, 2)], 0)
            out.to_csv(path, index=False); print("%s/%d done | alone %s" % (L, kf, " ".join("%s %.3f" % (k, np.sqrt(np.mean((out[k].values - y[vm]) ** 2))) for k in KINDS)), flush=True)
def summarize():
    src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3sum"}
    exec(src[:src.index("JUDGE = [")], g)
    T, B, C2, anch, LV, D1, CT, v3_for, rmf = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["D1"], g["CT"], g["v3_for"], g["rm"]
    rng = np.random.default_rng(20261225)
    def ld(L):
        fs = [f for f in os.listdir(CK) if f.startswith(L + "_")]; return pd.concat([pd.read_csv(os.path.join(CK, f)) for f in fs]).drop_duplicates("row_id")
    JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I"]; res = {k: dict(ok=True, diffs=[], lines=[], safety={}) for k in ("V3", "V3L", "V3T", "V30")}; vs = []
    for L in JUDGE + ["EXT10", "EXT12"]:
        d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
        X = M[["row_id"]].merge(ld(L), on="row_id", how="left"); cand = {"V3": v3_for(L, M), **{k: X[k].values for k in KINDS}}
        g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
        mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1))
        key = (M.farm + "_" + (M.day // 5).astype(str)).values; base, new = [], {k: [] for k in cand}
        for bi, b in enumerate(bs):
            for fam in "AB":
                a = B.w40(M, b, cx, fam); base.append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV))
                for k, v in cand.items():
                    miss = np.isnan(v); wgt = .1 * g_ * h_ * (~miss); a2 = a + wgt * (np.nan_to_num(v) - a); new[k].append(anch(M, C2.corrected(M, a2, D1, bi, fam, 7), LV))
        for k in cand:
            R = res[k]; ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(new[k], base)]
            R["lines"].append(f"{L:6s} missing {int(np.isnan(cand[k]).sum())} | all {' '.join('%+.2f' % (100 * x) for x in ch)} | alone {rmf(np.nan_to_num(cand[k], nan=np.nanmean(cand[k])) - y):.3f}")
            if L in JUDGE:
                R["ok"] &= all(x < 0 for x in ch)
                if L != "EL1": R["diffs"].append(pd.Series((new[k][0] - y) ** 2 - (base[0] - y) ** 2).groupby(key).mean())
            else:
                pw = []
                for n, b_ in zip(new[k], base):
                    dd = pd.Series((n - y) ** 2 - (b_ - y) ** 2).groupby(key).mean().values; pw.append(float((dd[rng.integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()))
                R["safety"][L] = not (np.mean(ch) > 0 and min(pw) < .025); R["ok"] &= R["safety"][L]
        if L in JUDGE and L != "EL1": vs.append(pd.Series((new["V3L"][0] - y) ** 2 - (new["V3"][0] - y) ** 2).groupby(key).mean())
        print("done", L, flush=True)
    for k, R in res.items():
        S = pd.concat(R["diffs"], axis=1).mean(axis=1).values; P = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
        print(f"\n=== {k} on v23 (vs no GRU) ==="); print("\n".join(R["lines"])); print(f"pooled P(worse) {P:.4f} | cold safety {R['safety']} -> {'qualifies' if (R['ok'] and P < .025) else 'does not qualify'}")
        if k == "V3L": verdict = R["ok"] and P < .025
    S = pd.concat(vs, axis=1).mean(axis=1).values; P = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
    print(f"\nV3L vs V3 (descriptive): pooled P(V3L worse) {P:.4f}")
    print("V3L:", "ADOPT (V3L qualifies to replace V3)" if verdict else "FAIL")
if __name__ == "__main__":
    run() if MODE == "cv" else summarize()
