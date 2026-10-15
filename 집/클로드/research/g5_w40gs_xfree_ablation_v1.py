# -*- coding: utf-8 -*-
"""G5 (2026-10-15 집 클로드; 사용자: 5점 감사 후 '1,2,3 순서로 진행' 의 2번).  서술 절제(제외 시 성능 보고용), 실행 전 커밋.  온도.
감사(설명자료_25_v2): 온도 기본 W40G-S 에 X 규칙 — BASE 의 원시 기록 번호 `day`, Codex·TabPFN 의 `second`(= day ≥ 179, 2차 구간 표시),
  그리고 기록 번호 순서로 이은 시계열에서 만든 '기록 넘는' 특징(시차·ewm·24/72/168시간 이동·d1/d3·dev6·경과 시간·물리 특징 시간 창).
세 판을 같은 코드로 다시 적합(TT1 과 같은 world·폴드·가중치·시드):
  REF = 지금 특징 그대로 (BASE ct, Codex FEATURE_COLUMNS day→season, PFN 같은 열)
  NX  = X 표시만 뺌: BASE 에서 day, Codex·PFN 에서 second (day→season 은 유지 = 날짜 정보, R)
  NXL = NX + 기록 넘는 특징 제거: ct 에서 이름에 _lag·_ewm·_r24·_r72·_r168·_d1·_d3·_dev6·hsince 가 있는 열, phc 에서 시간 창 있는 물리 열(ph_*_숫자),
        Codex·PFN 에서 reset1/3/8·delta 는 그날 안 계산인지 불명 → 보수적으로 같이 뺌
  PFN: TabPFN v2(world 의 저장 예측 대신 이 스크립트에서 세 판 모두 같은 절차로) — 문맥 = 학습 행 2000 무작위(rs = c), c ∈ 1..8 평균, n_estimators 4.
  W40G-S 꼴 = .4 BASE + (.2 + .4(1 − g)) CODEX + .4 g PFN,  g = clip((in_temp − 8)/2), BASE 시드 7·101.
검증기: TK2 world 의 DIAG10·EL1·EXT10·EXT12(TT1 과 같음).  계절 매핑 = world.mapping(학습 날 → 검증 날, 파일 안 씀).
보고: 판별 RMSE(전체·2차·추운 행), REF 대비 %.  판정 규칙 없음(5점 구성 후보 기본 모델의 '제외 시 손실' 서술).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u g5_w40gs_xfree_ablation_v1.py {cv|sum}
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, re
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"; sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"); sys.path.insert(0, TKDIR)
import world as W  # noqa: E402  (read-only import)
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "g5_ckpt"); SEEDS = (7, 101); CTX = range(1, 9)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
CROSS = re.compile(r"(_lag\d|_ewm\d|_r24|_r72|_r168|_d1$|_d3$|_dev6|hsince)")
def feature_sets(ct, phc, FC):
    ctNX = [c for c in ct if c != "day"]; fcNX = [c for c in FC if c != "second"]
    ctNXL = [c for c in ctNX if not CROSS.search(c)]; phNXL = [c for c in phc if not re.search(r"_\d+$", c) or c == "ph_farm_id"]
    fcNXL = [c for c in fcNX if not re.search(r"(reset\d|^delta$)", c)]
    return {"REF": (ct, phc, FC), "NX": (ctNX, phc, fcNX), "NXL": (ctNXL, phNXL, fcNXL)}
def season(tr, va, wv):
    td = tr[["farm", "day"]].drop_duplicates(); qd = va[["farm", "day"]].drop_duplicates()
    keys = set(td.itertuples(index=False, name=None)); v = {key: wv[key] for key in keys}
    sm, q, _ = W.mapping(td, qd, v); qm = dict(zip(qd.itertuples(index=False, name=None), q))
    a, b = tr.copy(), va.copy(); a["season"] = [sm[(f, int(d))] for f, d in zip(a.farm, a.day)]; b["season"] = [qm[(f, int(d))] for f, d in zip(b.farm, b.day)]
    return a, b
def pfn_pred(tr, va, cols):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    Xt = tr[cols].to_numpy(np.float32); yt = tr.sub_temp.to_numpy(float); Xv = va[cols].to_numpy(np.float32); out = []
    for c in CTX:
        i = np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=4, random_state=c,
                                                       ignore_pretraining_limits=True, inference_precision=torch.float32)
        m.fit(Xt[i], yt[i]); out.append(m.predict(Xv))
    return np.mean(out, 0)
def run():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); FC = list(W.FEATURE_COLUMNS); FS = feature_sets(ct, phc, FC)
    for k_, (c_, p_, f_) in FS.items(): print(k_, "base", len(c_), "phys", len(p_), "codex/pfn", len(f_), flush=True)
    print("NXL dropped from ct:", [c for c in ct if c not in FS["NXL"][0]], flush=True)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path): continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum(): continue
            tr, va = season(pfn[tm], pfn[vm], wv)
            out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); out["validator"], out["fold"] = name, k
            for tag, (cts, phs, fcs) in FS.items():
                for s in SEEDS: out["base_%s_%d" % (tag, s)] = W.base_predict(lab[tm], lab[vm], wb[tm], cts, phs, s)
                cols = [c if c != "day" else "season" for c in fcs]; save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
                try: out["codex_%s" % tag] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
                finally: W.TM.FEATURE_COLUMNS = save
                out["pfn_%s" % tag] = pfn_pred(tr.reset_index(drop=True), va.reset_index(drop=True), cols)
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)
def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1)); rows = []
    for tag in ("REF", "NX", "NXL"):
        for s in SEEDS: G[f"w_{tag}_{s}"] = .4 * G[f"base_{tag}_{s}"] + (.2 + .4 * (1 - g)) * G[f"codex_{tag}"] + .4 * g * G[f"pfn_{tag}"]
    for v in ("DIAG10", "EL1", "EXT10", "EXT12"):
        X = G[G.validator == v]; y = X.sub_temp.values; p2 = (X.day >= 179).values; cold = (X.in_temp <= 8).values
        for tag in ("REF", "NX", "NXL"):
            r = dict(val=v, arm=tag)
            for nm, m in (("all", np.ones(len(y), bool)), ("p2", p2), ("cold", cold)):
                r[nm] = np.mean([rm((X[f"w_{tag}_{s}"].values - y)[m]) for s in SEEDS]) if m.any() else np.nan
            for mem in ("base", "codex", "pfn"):
                col = f"{mem}_{tag}_7" if mem == "base" else f"{mem}_{tag}"; r[mem] = rm(X[col].values - y)
            rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
    for v in ("DIAG10", "EL1", "EXT10", "EXT12"):
        a = R[(R.val == v)].set_index("arm"); print(v, {t: f"{100 * (a.at[t, 'all'] / a.at['REF', 'all'] - 1):+.1f}% (p2 {100 * (a.at[t, 'p2'] / a.at['REF', 'p2'] - 1):+.1f}%)" for t in ("NX", "NXL")})
if __name__ == "__main__":
    run() if MODE == "cv" else summarize()
