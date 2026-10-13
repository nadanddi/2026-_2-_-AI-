# -*- coding: utf-8 -*-
"""SV3a (2026-10-14 집 클로드; 사용자: '올바른 검증기로 내 모델을 검증').  검증 기반 — SV1/SV2 비평(6.851) 반영 판: 배치 생성 + 층 표시 + 배치별 실제 파이프라인 예측 저장.
배치(평가 구조 흉내, 6.850): 2차 정답 기록(잠금 제외)의 연속 구간 중 길이 >= 3 인 것마다 배치 1개 — 그 구간의 양끝 = del(평가 덩어리 앞뒤 삭제처럼), 가운데 = val.
  나머지 2차 구간은 모두 보임(평가처럼 이웃).  같은 날 다른 ID 정답 기록은 확률 .28 로 tw(정답·입력 모두 안 씀; 배치마다 고정 난수).
  잠금 40일: 어디서도 안 보임(학습·계절 기준·SG2 참조·층 표시 모두에서 제외; 채점도 안 함 — lab 에 없음).
학습 = 정답 있음 & 잠금 아님 & val·del·tw 아님(구조 밖 추가 제외 없음 — 평가와 같게).  계절 지표 기준 날 = 그 학습 날.
파이프라인(실제, CT2 = submission_14 형): pre = clip(shrink(.8 R3 + .2 PFN)), R3 = .6 ET + .3 LGB + .1 MLP(DC4 + DP1), PFN = TabPFN v2 문맥 2000행 x 4(새 문맥 17~20),
  fin = SG2 보정(참조 = 그 배치의 보이는 정답 날) 후 clip.  새 시드 2323·4545·6767.
층(후보와 무관): prev_vis = cert 진짜 전날이 그 배치에서 보이는 정답 기록;  twin_vis = 학습 달력 같은 날짜·다른 ID 보이는 정답 기록 있음;  high = 실제 하루 평균 >= 1.0(서술용).
저장: local/sv3_ckpt/draw_<k>.csv (val 행: rec, hour, y, pre_s, fin_s), sv3_draws_v1.csv(draw, rec, role), sv3_strata_v1.csv.
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x", "1"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
wt0, dc4, p3, core, sg2 = ct2.wt0, ct2.dc4, ct2.p3, ct2.core, ct2.sg2
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
SEEDS = (2323, 4545, 6767); CTX = (17, 18, 19, 20); W = (.6, .3, .1)
CK = os.path.join(env.LOCAL, "sv3_ckpt"); os.makedirs(CK, exist_ok=True)
key = lambda f, d: f"{f}_{int(d):03d}"
raw, full, lab, lock, fds = ct2.prep(); wv = dc4.weather_vectors(full); cols, PF = ct2.feature_sets(); FS, BS = cols["REF"]
lab["rec"] = [key(f, d) for f, d in zip(lab.farm, lab.day)]
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
LOCKREC = {key(f, d) for f, d in lockd}
LABREC = set(lab.rec) - LOCKREC
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s_, d_) for r, s_, d_ in zip(C.rec, C.seg, C.day)}
def runs(nums):
    out, cur = [], []
    for n in sorted(nums):
        if cur and n == cur[-1] + 1: cur.append(n)
        else:
            if cur: out.append(cur)
            cur = [n]
    if cur: out.append(cur)
    return out
DRAWS = []
for f in ("F13", "F47"):
    p2 = [int(r[4:]) for r in LABREC if r[:3] == f and int(r[4:]) >= 179]
    for rn in runs(p2):
        if len(rn) >= 3: DRAWS.append((f, rn))
print("draws:", len(DRAWS), [(f, rn[0], rn[-1]) for f, rn in DRAWS], flush=True)
R, WV, hrs, SIG = sg2.prepare_structure()
ec = lab.groupby(["farm", "day"]).sub_ec.mean()
drows, srows = [], []
for k, (f, rn) in enumerate(DRAWS):
    rng = np.random.default_rng(7000 + k); role = {key(f, rn[0]): "del", key(f, rn[-1]): "del"}
    for n in rn[1:-1]: role[key(f, n)] = "val"
    for r in [x for x, v in role.items() if v == "val"]:
        if r not in DAY: continue
        for o in LABREC:
            if o[:3] != f and o not in role and DAY.get(o) == DAY[r] and rng.random() < .28: role[o] = "tw"
    for r, v in role.items(): drows.append(dict(draw=k, rec=r, role=v))
    vis = LABREC - set(role)
    for r in [x for x, v in role.items() if v == "val"]:
        tw = r in DAY and any(o[:3] != r[:3] and DAY.get(o) == DAY[r] for o in vis)
        srows.append(dict(draw=k, rec=r, prev_vis=(r in CERT and CERT[r] in vis), twin_vis=bool(tw)))
    path = os.path.join(CK, "draw_%d.csv" % k)
    if os.path.exists(path): print("draw", k, "cached", flush=True); continue
    tr = lab[lab.rec.isin(vis)].copy(); va = lab[lab.rec.isin([x for x, v in role.items() if v == "val"])].copy()
    tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr["season"] = [season[(a, d)] for a, d in zip(tr.farm, tr.day)]; va = va.merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
    lo, hi = tr.sub_ec.min(), tr.sub_ec.max(); Xt, yt, Xv = tr[PF].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[PF].to_numpy(np.float32)
    pf = np.mean([ct2.pfn(Xt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], yt[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], Xv, c) for c in CTX], axis=0)
    G = va[["row_id", "rec", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True); vaR = va.reset_index(drop=True)
    vd = {(a, int(d)) for a, d in zip(va.farm, va.day)}; ref = {(a, int(d)) for a, d in zip(tr.farm, tr.day)}; cal = sg2.ref_calendar(R, WV, ref)
    for s in SEEDS:
        e, l, m = wt0.members(tr, va, s, FS, BS); r3 = W[0] * e + W[1] * l + W[2] * m
        G["pre_%d" % s] = np.clip(ct2.shrink(.8 * r3 + .2 * pf, vaR), lo, hi)
        G["fin_%d" % s] = np.clip(sg2.correction(G, "pre_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
    G["draw"] = k; G.to_csv(path, index=False); print("draw", k, f, rn[0], "-", rn[-1], "val", len(vd), "tw", sum(v == "tw" for v in role.values()), "done", flush=True)
pd.DataFrame(drows).to_csv(os.path.join(HERE, "sv3_draws_v1.csv"), index=False); pd.DataFrame(srows).to_csv(os.path.join(HERE, "sv3_strata_v1.csv"), index=False)
S = pd.DataFrame(srows); print("val records:", len(S), "| prev_vis", int(S.prev_vis.sum()), "| twin_vis", int(S.twin_vis.sum()))
