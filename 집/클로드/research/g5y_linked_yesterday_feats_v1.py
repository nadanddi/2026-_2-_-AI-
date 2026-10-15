# -*- coding: utf-8 -*-
"""G5Y (2026-10-15 집 클로드; 사용자: 26번 설계 → '2,3번 진행' 의 2번 — 번호 순서 시계열 특징을 '링크된 같은 동 어제' 특징으로 대체).  서술 절제, 실행 전 커밋.  온도.
G5 의 NXL(X 표시 + 기록 넘는 특징 제거) BASE 에 '링크된 어제' 특징 16개를 더한 NXY 를 같은 폴드로 다시 적합(BASE 만; Codex·PFN 은 G5 NXL 예측 그대로).
어제 = 패키지 25 MASK 세계 거울 링커 prev0, p0 ≥ .5 (v15r_mirror_masked_pkg25.csv; 26번 학습 행 연결과 같은 규칙), 어제가 학습 기록일 때만(평가 기록이면 MASK 세계라 NaN).
특징: y_has, y_in23·y_in22·y_in21, y_in_mean·min·max, y_out_mean, y_rad_mean(rad_eff), y_heat_mean, y_hum_mean,
      [어제 24시간 + 오늘 0..h] 이은 창에서 in_temp ewm(반감 2·6), in_temp 24시간 평균, rad_eff ewm6, out_temp ewm6 (그 시각 값).
비교: REF / NX / NXL / NXY 의 W40G-S 꼴(.4 BASE + (.2 + .4(1−g)) CODEX + .4 g PFN; NXY 의 CODEX·PFN = NXL 것), DIAG10·EL1·EXT10·EXT12.
보고: RMSE 전체·2차·추운 행, NXL 대비 NXY 회복분(= NX 와의 차이 몇 % 메웠나).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u g5y_linked_yesterday_feats_v1.py {cv|sum}   (G5 체크포인트 필요)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "g5_w40gs_xfree_ablation_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "g5_w40gs_xfree_ablation_v1.py")
exec(src[:src.index("def run():")])
__file__ = os.path.join(HERE, "g5y_linked_yesterday_feats_v1.py")
CKY = os.path.join(env.LOCAL, "g5y_ckpt")
YF = ["y_has", "y_in23", "y_in22", "y_in21", "y_in_mean", "y_in_min", "y_in_max", "y_out_mean", "y_rad_mean", "y_heat_mean", "y_hum_mean",
      "l48_in_ewm2", "l48_in_ewm6", "l48_in_r24", "l48_rad_ewm6", "l48_out_ewm6"]
def ewm_last(x, hl):
    a = 1 - 0.5 ** (1 / hl); m, w = 0.0, 0.0
    for v in x:
        if np.isfinite(v): m = (1 - a) * m + a * v; w = (1 - a) * w + a
        else: m, w = (1 - a) * m, (1 - a) * w
    return m / w if w > 0 else np.nan
def add_yfeats(lab):
    L = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); link = {r: p for r, p, q in zip(L.rec, L.prev0, L.p0) if isinstance(p, str) and q >= .5 and p != r}
    lab = lab.copy(); lab["rec"] = lab.farm + "_" + lab.day.astype(int).map("{:03d}".format)
    cols = {"in": "in_temp", "out": "out_temp", "rad": "rad_eff", "heat": "act_heating", "hum": "in_hum"}; DAY = {}
    for r, g in lab.groupby("rec"):
        m = {k: np.full(24, np.nan) for k in cols}; hh = g.hour.values.astype(int)
        for k, c in cols.items(): m[k][hh] = g[c].values
        DAY[r] = m
    F = {c: np.full(len(lab), np.nan) for c in YF}
    for j, (r, h) in enumerate(zip(lab.rec.values, lab.hour.values.astype(int))):
        p = link.get(r); Y = DAY.get(p) if p else None; T_ = DAY[r]
        F["y_has"][j] = float(Y is not None)
        if Y is not None:
            F["y_in23"][j], F["y_in22"][j], F["y_in21"][j] = Y["in"][23], Y["in"][22], Y["in"][21]
            with np.errstate(all="ignore"):
                F["y_in_mean"][j], F["y_in_min"][j], F["y_in_max"][j] = np.nanmean(Y["in"]), np.nanmin(Y["in"]), np.nanmax(Y["in"])
                F["y_out_mean"][j], F["y_rad_mean"][j], F["y_heat_mean"][j], F["y_hum_mean"][j] = np.nanmean(Y["out"]), np.nanmean(Y["rad"]), np.nanmean(Y["heat"]), np.nanmean(Y["hum"])
        cat = lambda k: np.concatenate([Y[k] if Y is not None else np.full(24, np.nan), T_[k][:h + 1]])
        ci = cat("in"); F["l48_in_ewm2"][j] = ewm_last(ci, 2); F["l48_in_ewm6"][j] = ewm_last(ci, 6)
        with np.errstate(all="ignore"): F["l48_in_r24"][j] = np.nanmean(ci[-24:])
        F["l48_rad_ewm6"][j] = ewm_last(cat("rad"), 6); F["l48_out_ewm6"][j] = ewm_last(cat("out"), 6)
    for c in YF: lab[c] = F[c]
    return lab
def run():
    os.makedirs(CKY, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); FS = feature_sets(ct, phc, list(W.FEATURE_COLUMNS)); lab = add_yfeats(lab)
    ctY = FS["NXL"][0] + YF; phY = FS["NXL"][1]; print("NXY base features", len(ctY), "| rows with linked yesterday", round(float(lab.y_has.mean()), 3), flush=True)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CKY, "%s_%d.csv" % (name, k))
            if os.path.exists(path): continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum(): continue
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True)
            for s in SEEDS: out["base_NXY_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ctY, phY, s)
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)
def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    Y = pd.concat([pd.read_csv(os.path.join(CKY, f)) for f in sorted(os.listdir(CKY))], ignore_index=True)
    G = G.merge(Y, on="row_id", how="left"); G["codex_NXY"] = G.codex_NXL; G["pfn_NXY"] = G.pfn_NXL
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1)); rows = []
    for tag in ("REF", "NX", "NXL", "NXY"):
        for s in SEEDS: G[f"w_{tag}_{s}"] = .4 * G[f"base_{tag}_{s}"] + (.2 + .4 * (1 - g)) * G[f"codex_{tag}"] + .4 * g * G[f"pfn_{tag}"]
    for v in ("DIAG10", "EL1", "EXT10", "EXT12"):
        X = G[G.validator == v]; y = X.sub_temp.values; p2 = (X.day >= 179).values; cold = (X.in_temp <= 8).values
        for tag in ("REF", "NX", "NXL", "NXY"):
            r = dict(val=v, arm=tag)
            for nm, m in (("all", np.ones(len(y), bool)), ("p2", p2), ("cold", cold)):
                r[nm] = np.mean([rm((X[f"w_{tag}_{s}"].values - y)[m]) for s in SEEDS]) if m.any() else np.nan
            r["base_alone"] = rm(X[f"base_{tag}_7"].values - y); rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
    for v in ("DIAG10", "EL1", "EXT10", "EXT12"):
        a = R[R.val == v].set_index("arm"); gap = a.at["NXL", "all"] - a.at["NX", "all"]
        print(v, {t: f"{100 * (a.at[t, 'all'] / a.at['REF', 'all'] - 1):+.1f}%" for t in ("NX", "NXL", "NXY")}, f"| NXY recovers {100 * (a.at['NXL', 'all'] - a.at['NXY', 'all']) / gap:.0f}% of the NXL−NX gap" if abs(gap) > 1e-9 else "")
if __name__ == "__main__":
    run() if MODE == "cv" else summarize()
