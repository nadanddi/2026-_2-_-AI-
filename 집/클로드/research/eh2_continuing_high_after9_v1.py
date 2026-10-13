# -*- coding: utf-8 -*-
"""EH2 (2026-10-14 집 클로드; 사용자: '1번(이른 시각 엄격)이 낫지 않을까').  사전 고정(커밋 후 실행).  EC.  과거 정보만.  2차만.
EH1(6.864) 사후 관찰: 가짜 발동(F47_212·214·216)은 모두 0~6시, 진짜(F13_243)는 하루 내내, 평가 F13_221 은 8시 이후.
수정 하나: EH1 규칙을 **9시 이후(h >= 9)에만** 적용 — 9시 = 고EC 동 운영 특징(커튼 9시 닫힘, CO2 9·11·13시)이 드러나는 시각(HS2 6.690; 이 결과와 별개 근거).
  [사후 설계 명시: 문턱을 EH1 결과를 본 뒤 정함 → EH1 에 쓴 검증기(DIAG10·P2LOO·EL1·EV)로는 판정하지 않음]
판정 배치 = TL1 의 새 배치 EL4o2(11폴드, 실제 파이프라인 REF fin, 새 시드 2121·3434·5656) — EH1 설계에 안 쓰임.  기준 = REF fin(앵커 없음).
  보이는 정답 = 학습 & 그 폴드 학습 제외(검증 날·같은 농장 ±1·다른 농장 −3..+3·잠금 ±1, TL1 과 같음) 아님.
판정: 3시드 모두 EL4o2 2차 dSSE < 0 AND (날 단위 부트스트랩 20,000 P(낫지 않음) < .025 OR 나빠진 날 없음).
서술: 같은 수정의 원래 검증기 결과(사후 — 참고만), 평가 발동 기록.
"""
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh1_continuing_high_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh1_continuing_high_lift_v1.py")
exec(src[:src.index("res, recd, blocks, fl = [], [], {}, []")])
__file__ = os.path.join(HERE, "eh2_continuing_high_after9_v1.py"); H0 = 9
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
LOCKD = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]}
OTHER = {"F13": "F47", "F47": "F13"}
D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(env.LOCAL, "tl1_ckpt", "EL4o2_*.csv"))], ignore_index=True)
D = D[D.day >= 179].sort_values(["validation_fold", "farm", "day", "hour"]).reset_index(drop=True); D["rec"] = [f"{f}_{int(d):03d}" for f, d in zip(D.farm, D.day)]
SEEDS = (2121, 3434, 5656); rows = []; fired = {}
for k, g in D.groupby("validation_fold"):
    vd = {(f, int(d)) for f, d in zip(g.farm, g.day)}
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)} | {(f, d + j) for f, d in LOCKD for j in (-1, 0, 1)}
    vis = TRAIN & np.isfinite(DM) & ~np.array([(FARM[i], int(NUM[i])) in forb for i in range(N)])
    for r, gr in g.groupby("rec"):
        f, pick = fires(idx[r], vis); f = [x and h >= H0 for h, x in enumerate(f)]
        fired[r] = (sum(f), rid(pick) if pick is not None and any(f) else None, bool(DM[idx[r]] >= 1.0))
        for s in SEEDS:
            p0 = gr[f"fin_REF_{s}"].values; p1 = np.where([f[h] for h in gr.hour.values], np.clip(p0 + LIFT, lo_, hi_), p0); y = gr.sub_ec.values
            rows.append(dict(seed=s, rec=r, farm=r[:3], day=int(r[4:]), d=float(np.sum((p1 - y) ** 2) - np.sum((p0 - y) ** 2)), sse0=float(np.sum((p0 - y) ** 2)), n=len(y)))
    print("fold", k, "done", flush=True)
Rr = pd.DataFrame(rows)
tot = Rr.groupby("seed").agg(d=("d", "sum"), sse0=("sse0", "sum"), n=("n", "sum")); tot["pct"] = 100 * (np.sqrt((tot.sse0 + tot.d) / tot.n) / np.sqrt(tot.sse0 / tot.n) - 1)
print(tot.round(4).to_string())
per_day = Rr.groupby("rec").d.mean(); v = per_day.values; ix = np.random.default_rng(20261212).integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean())
harm = per_day[per_day > 1e-6]; ok = bool((tot.d < 0).all()) and (P < .025 or len(harm) == 0)
print("fired records:", {r: x for r, x in fired.items() if x[0] > 0})
print(f"EL4o2: seeds better {int((tot.d < 0).sum())}/3 | day-bootstrap P(not better) {P:.4f} | harmed days {len(harm)} {harm.round(3).to_dict()} -> {'ADOPT' if ok else 'FAIL'}")
print("EH2:", "ADOPT" if ok else "FAIL")
visE = TRAIN & np.isfinite(DM); print("\neval records where EH2 fires (descriptive):")
for i in np.where(~TRAIN)[0]:
    if NUM[i] < 179: continue
    f, pick = fires(i, visE); f = [x and h >= H0 for h, x in enumerate(f)]
    if any(f): print(f"   {rid(i)}: hours {sum(f)}/24 (from {min(h for h in range(24) if f[h])}h), ancestor {rid(pick)} (EC {DM[pick]:.2f})")
