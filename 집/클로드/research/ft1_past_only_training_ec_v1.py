# -*- coding: utf-8 -*-
"""FT1 (2026-10-14 집 클로드; 사용자: 5점 모델 '현장 정보 해석' 결정 → 근거 시험 C2 '과거만 학습' 을 연구실 GPU 로).  서술 측정(채택 판정 아님), 실행 전 커밋.  EC.
질문: 현장 모델은 대상 날짜보다 '앞선' 날의 정답만 학습할 수 있다(미래 정답 없음).  지금 EC 기본 파이프라인의 성능, 특히 계절(날짜) 정보의 이득이
  '앞뒤 이웃 날짜 정답 빌리기' 에서 오는지(SZ2 해석), 과거만 학습해도 남는지 측정.
파이프라인: SZ2·CT2 와 같음 — pre = clip(shrink(.8 R3 + .2 PFN)), R3 = .6 ET + .3 LGB + .1 MLP(DP1 운영 특징), PFN = TabPFN v2 문맥 2000행 x 4.  SG2 없음(대체 예정).
날짜(CAL): 현장에선 달력 날짜를 앎 → 정답 있는 F13·F47 모든 날에 대해 dc4.season_index 로 한 번 추정(1차 = 기록 번호, 2차 = 외기 일치 날짜)한 값을
  '알려진 날짜' 로 고정해 계절 특징과 과거 판정에 모두 씀(폴드마다 다시 추정하지 않음).
대상 날(target): 2차 정답 날 전부(번호 ≥ 179) + 1차 날 중 CAL ≥ 60 인 날을 농장별 번호 순 6번째마다.  한 대상 날 = 한 폴드.
학습 배치 2개 x 계절 2개 = 4안:
  PAST = F13·F47 정답 날 중 CAL < 대상 CAL (같은 날짜 제외, 두 농장 모두) − 잠금 ±1
  ALL  = SZ2 숨김(같은 농장 번호 ±1, 다른 농장 −3..+3, 잠금 ±1) 외 전부(미래 날짜 포함)
  DATE = R3·PFN 입력에 season = CAL,  NOS = season 뺌
  시드 R3 1357·2468·3579, PFN 문맥 21~24 (SZ2 와 같음).
보고(sum): 안별 RMSE(전체 / 2차 / 1차 / 보통 / 고EC), 날짜 이득(NOS − DATE) 을 PAST·ALL 에서 비교, PAST 의 대상 CAL 구간별(60~90 / 90~120 / 120~).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ft1_past_only_training_ec_v1.py {list|cv|sum}
  list : 대상 날·학습 크기만 출력(학습 안 함)
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json, glob
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "list"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "1"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
wt0, dc4 = ct2.wt0, ct2.dc4
SEEDS = (1357, 2468, 3579); CTX = (21, 22, 23, 24); W = (.6, .3, .1); OTHER = {"F13": "F47", "F47": "F13"}
CK = os.path.join(env.LOCAL, "ft1_ckpt"); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def setup():
    raw, full, lab, lock, fds = ct2.prep(); wv = dc4.weather_vectors(full); cols, PF = ct2.feature_sets(); FS, BS = cols["REF"]
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    days = lab[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    cal, _ = dc4.season_index(days, days.iloc[:0], wv); CAL = {(f, int(d)): float(cal[(f, d)]) for f, d in zip(days.farm, days.day)}
    lab = lab.copy(); lab["season"] = [CAL[(f, int(d))] for f, d in zip(lab.farm, lab.day)]
    tg = []
    for f in ("F13", "F47"):
        p2 = sorted(d for (ff, d) in CAL if ff == f and d >= 179)
        p1 = [d for d in sorted(d for (ff, d) in CAL if ff == f and d < 179) if CAL[(f, d)] >= 60][::6]
        tg += [(f, d) for d in p1 + p2]
    tg = [t for t in tg if t not in lockd]
    return lab, lockd, CAL, tg, FS, BS, PF
def masks(lab, lockd, CAL, t):
    f, d = t; c = CAL[t]
    lk = {(ff, dd + j) for ff, dd in lockd for j in (-1, 0, 1)}
    key = list(zip(lab.farm, lab.day.astype(int)))
    past = np.array([CAL[k] < c and k not in lk for k in key])
    forb = {(f, d + j) for j in (-1, 0, 1)} | {(OTHER[f], d + j) for j in range(-3, 4)} | lk
    allm = np.array([k not in forb for k in key])
    va = np.array([k == t for k in key])
    return past, allm, va
def run(mode):
    lab, lockd, CAL, tg, FS, BS, PF = setup(); print("targets", len(tg), "(pass-2", sum(d >= 179 for _, d in tg), ")", flush=True)
    if mode == "list":
        for t in tg:
            past, allm, va = masks(lab, lockd, CAL, t); print(t, "CAL %.1f" % CAL[t], "train days PAST", lab[past][["farm", "day"]].drop_duplicates().shape[0], "ALL", lab[allm][["farm", "day"]].drop_duplicates().shape[0], "rows", int(va.sum()))
        return
    os.makedirs(CK, exist_ok=True)
    ARMS = {"DATE": (FS, BS, PF), "NOS": ([c for c in FS if c != "season"], [c for c in BS if c != "season"], [c for c in PF if c != "season"])}
    for t in tg:
        path = os.path.join(CK, "%s_%03d.csv" % t)
        if os.path.exists(path): continue
        past, allm, va = masks(lab, lockd, CAL, t); vaD = lab[va].copy(); vaR = vaD.reset_index(drop=True)
        G = vaD[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True); G["cal"] = CAL[t]
        for reg, m in (("PAST", past), ("ALL", allm)):
            tr = lab[m].copy(); lo, hi = tr.sub_ec.min(), tr.sub_ec.max(); G[f"ntrain_{reg}"] = tr[["farm", "day"]].drop_duplicates().shape[0]
            for arm, (fs, bs, pf) in ARMS.items():
                Xt, yt, Xv = tr[pf].to_numpy(np.float32), tr.sub_ec.to_numpy(float), vaD[pf].to_numpy(np.float32)
                pfp = np.mean([ct2.pfn(Xt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], yt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], Xv, c) for c in CTX], axis=0)
                for s in SEEDS:
                    e, l, mm = wt0.members(tr, vaD, s, fs, bs); r3 = W[0] * e + W[1] * l + W[2] * mm
                    G[f"{reg}_{arm}_{s}"] = np.clip(ct2.shrink(.8 * r3 + .2 * pfp, vaR), lo, hi)
        G.to_csv(path, index=False); print("%s_%03d done" % t, flush=True)
def summarize():
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CK, "*.csv"))], ignore_index=True)
    D["hi"] = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1.0; D["p2"] = D.day >= 179; y = D.sub_ec.values
    print("target days", D.groupby(["farm", "day"]).ngroups, "rows", len(D), "| pass-2 days", D[D.p2].groupby(["farm", "day"]).ngroups, "high days", D[D.hi].groupby(["farm", "day"]).ngroups)
    sc = {"all": np.ones(len(D), bool), "p2": D.p2.values, "p1": ~D.p2.values, "normal": ~D.hi.values, "high": D.hi.values,
          "cal<60": (D.cal < 60).values, "cal60_90": ((D.cal >= 60) & (D.cal < 90)).values, "cal90_120": ((D.cal >= 90) & (D.cal < 120)).values, "cal120+": (D.cal >= 120).values}
    rows = []
    for reg in ("PAST", "ALL"):
        for arm in ("DATE", "NOS"):
            r = dict(arm=f"{reg}_{arm}")
            for k, m in sc.items(): r[k] = np.mean([rm((D[f"{reg}_{arm}_{s}"].values - y)[m]) for s in SEEDS]) if m.any() else np.nan
            rows.append(r)
    R = pd.DataFrame(rows).set_index("arm"); pd.set_option("display.width", 220); print(R.round(4).to_string())
    for reg in ("PAST", "ALL"):
        print(f"date gain in {reg} (NOS/DATE − 1, %):", (100 * (R.loc[f"{reg}_NOS"] / R.loc[f"{reg}_DATE"] - 1)).round(1).to_dict())
    print("PAST vs ALL (DATE, %):", (100 * (R.loc["PAST_DATE"] / R.loc["ALL_DATE"] - 1)).round(1).to_dict())
    print("train days PAST median", float(D.groupby(["farm", "day"]).ntrain_PAST.first().median()), "ALL median", float(D.groupby(["farm", "day"]).ntrain_ALL.first().median()))
if __name__ == "__main__":
    summarize() if MODE == "sum" else run(MODE)
