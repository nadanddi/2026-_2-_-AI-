# ECL3 (2026-10-13 집 클로드; 사용자: '평가 쪽도 확인 불가 앵커가 많지 않나') 서술: EC 변형(v7 ∪ v12 p≥.6 & HP2 일치)을 평가 60 에 적용 — 새 앵커가
# NA4(6.703) 분류의 '이을 수 없는 기록'(N1 전날 날짜에 정답 없음, N2 전날로 같은 ID 평가 기록을 고름)에 붙는지. 학습 전용 달력(DMT2) 기준 날짜 상태도.
import os, sys
import env
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])
PEV = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_eval.csv")).set_index("rec")
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
NA4 = pd.read_csv(os.path.join(HERE, "na4_v1.csv")).set_index("rec")
rows = []
for r, row in PEV.iterrows():
    if r in v7e: continue
    base = dict(rec=r, prev=row.prev, p=round(row.p, 3), dT=row.dT, date=row.date_status, na4=NA4.cat.get(r, "(anchored by v7)"))
    if not row.labelled or row.p < .6 or row.date_status == "mismatch": base["decision"] = "no (conf/date)"
    else:
        h = hp2(idx[r], TRAIN.copy()); base["hp2"] = h; base["decision"] = "ANCHOR" if h == row.prev else "no (HP2 disagrees)"
    rows.append(base)
T = pd.DataFrame(rows); pd.set_option("display.width", 220)
A = T[T.decision == "ANCHOR"]
print(f"eval: v7 anchors {len(v7e)} | EC-variant new anchors {len(A)}")
print(A[["rec", "prev", "p", "dT", "date", "na4", "hp2"]].to_string(index=False))
print("\nnew anchors by NA4 category:", A.na4.value_counts().to_dict())
print("records in unlinkable NA4 categories (N1/N2) that got an anchor:", A[A.na4.str.startswith(("N1", "N2"))].rec.tolist())
print("\nwhy others not anchored:", T[T.decision != "ANCHOR"].decision.value_counts().to_dict())
