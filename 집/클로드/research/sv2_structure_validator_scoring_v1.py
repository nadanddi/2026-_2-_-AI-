# -*- coding: utf-8 -*-
"""SV2 (2026-10-14 집 클로드; 사용자: '2단계까지 추가하고 비평가 붙여 평가').  검증 기반 2단계: SV1 배치(20개)에 예측을 꽂아 상황별 채점 + 검증기 자체 점검(양성·음성 대조).
기본 예측 BASE = 대리 기본 모델(R3 의 ET 구성원 core.et 시드 47 + DP1 + DC4 계절, p3.final) — 배치마다 학습 = 정답 있음 & val·del·tw 아님.  잠금 40일은 lab 에 없어 채점 제외.
후보(모두 EA2 꼴: P_h = BASE_h + (A − cm_h), cm_h = BASE 0..h 누적평균):
  ORA = 양성 대조: cert 진짜 전날이 그 배치에서 보이는 정답이면 A = 그 23시 EC (1·2차 모두).  '쓸 수 없는 정보' — 검증기가 개선을 잡아내는지 보는 용도.
  SHUF = 음성 대조: ORA 와 같은 기록에, A = 같은 농장의 보이는 정답 기록 중 무작위 하나의 23시 EC(배치·기록마다 고정 난수).
  LKH = 실제 후보: 2차 val 중 시각별 지문 링커(SV1 의 LKH, h 시 0..h 지문, 확신·tau 조건)가 짝을 찾은 시각에 A = 짝 23시 EC (LH3 P2 와 같은 규칙).
채점: 상황(S1 짝 찾음 / S2 짝 있으나 못 찾음 / S3 짝 없음) x (2차/1차) x (보통/고EC)별 dSSE 와 RMSE 변화; 배치 단위 '개선 비율'(20개 중), 기록 단위 부트스트랩 P(악화).
상황별 판정 틀(검증기 규칙 초안 — 이 파일에선 대조로 '규칙이 제대로 가르는지' 만 봄): 목표 상황에서 배치 개선 비율 >= .9 AND 기록 부트스트랩 P(악화) < .025, 다른 상황은 유의 악화 없음.
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc4, p3, core = dp1.dc4, dp1.p3, dp1.core
raw, full, lab, lock, signatures, fds = p3.prepare()
lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])])); wv = dc4.weather_vectors(full)
FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
SV1 = open(os.path.join(HERE, "sv1_structure_validator_v1.py"), encoding="utf-8").read(); ns = {"__file__": os.path.join(HERE, "sv1_structure_validator_v1.py"), "__name__": "sv1"}
exec(SV1[:SV1.index("D, S = [], []")], ns)
EC, CERT, LAB, idx, N, FARM, NUM, TAU, dist, tdate, DAYT, SEGT = (ns[k] for k in ("EC", "CERT", "LAB", "idx", "N", "FARM", "NUM", "TAU", "dist", "tdate", "DAYT", "SEGT"))
rid = ns["rid"]; DM = ns["DM"]
DR = pd.read_csv(os.path.join(HERE, "sv1_draws_v1.csv")); SI = pd.read_csv(os.path.join(HERE, "sv1_situations_v1.csv"))
key = lambda f, d: f"{f}_{int(d):03d}"
lab["rec"] = [key(f, d) for f, d in zip(lab.farm, lab.day)]
lo_, hi_ = lab.sub_ec.min(), lab.sub_ec.max(); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def lkh_picks(b, vis):
    t = tdate(b, vis)
    if t is None: return [None] * 24
    cand = [a for a in np.where(vis)[0] if a != b and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]]
    out = []
    for h in range(24):
        if not cand: out.append(None); continue
        d = sorted((dist(a, b, h), a) for a in cand); ok = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= TAU[h]
        out.append(d[0][1] if ok else None)
    return out
rows = []
for k, g in DR.groupby("draw"):
    hid = set(g.rec); val = set(g[g.role == "val"].rec)
    vis = LAB & ~np.array([rid(i) in hid for i in range(N)])
    tr = lab[~lab.rec.isin(hid)].copy(); va = lab[lab.rec.isin(val)].copy()
    tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; va = va.merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
    va = va.sort_values(["rec", "hour"]); va["base"] = p3.final(core.predict_model(core.et(47), tr, va, FS), tr, va)
    va["cm"] = va.groupby("rec").base.transform(lambda s: s.expanding().mean())
    rng = np.random.default_rng(5000 + k); visf = {f: [i for i in np.where(vis)[0] if FARM[i] == f] for f in ("F13", "F47")}
    for r, G in va.groupby("rec"):
        b = idx[r]; y = G.sub_ec.values; base = G.base.values; cm = G.cm.values; hh = G.hour.values
        ora = shuf = lk = base.copy()
        if b in CERT and vis[CERT[b]]:
            A = EC[CERT[b], 23]; ora = np.clip(base + (A - cm), lo_, hi_)
            As = EC[int(rng.choice(visf[FARM[b]])), 23]; shuf = np.clip(base + (As - cm), lo_, hi_)
        if NUM[b] >= 179:
            pk = lkh_picks(b, vis); lk = base.copy()
            for j, h in enumerate(hh):
                if pk[h] is not None: lk[j] = np.clip(base[j] + (EC[pk[h], 23] - cm[j]), lo_, hi_)
        sse = lambda p: float(np.sum((p - y) ** 2))
        rows.append(dict(draw=k, rec=r, base=sse(base), ORA=sse(ora), SHUF=sse(shuf), LKH=sse(lk), n=len(y)))
    print("draw", k, "done", flush=True)
R = pd.DataFrame(rows).merge(SI, on=["draw", "rec"], how="left")
R["sit"] = np.select([R.S1, R.S2], ["S1 found", "S2 pair-not-found"], "S3 no-pair"); R["pass"] = np.where(R.pass2, "p2", "p1")
R.to_csv(os.path.join(HERE, "sv2_record_sse_v1.csv"), index=False)
rng = np.random.default_rng(20261208); pd.set_option("display.width", 220); out = []
for (ps, sit, hi), g in R.groupby(["pass", "sit", "hi"]):
    for c in ("ORA", "SHUF", "LKH"):
        d = (g[c] - g.base); per_draw = d.groupby(g.draw).sum(); rec = d.groupby(g.rec).mean().values
        P = float((rec[rng.integers(0, len(rec), (20000, len(rec)))].sum(1) >= 0).mean()) if len(rec) > 1 else np.nan
        out.append(dict(pass_=ps, sit=sit, high=hi, cand=c, recs=g.rec.nunique(), rmse_base=np.sqrt(g.base.sum() / g.n.sum()), pct=100 * (np.sqrt(g[c].sum() / g.n.sum()) / np.sqrt(g.base.sum() / g.n.sum()) - 1),
                        draws_better=float((per_draw < -1e-9).mean()), draws_changed=float((per_draw.abs() > 1e-9).mean()), P_worse=P))
O = pd.DataFrame(out); print(O.round(3).to_string(index=False))
O.to_csv(os.path.join(HERE, "sv2_situation_table_v1.csv"), index=False)
