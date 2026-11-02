"""XID1: '같은 동이 두 온실 ID를 오간다 — 확실한 짝의 약 4분의 1'(정형 README ④, 덱 1.1) 출처 재계산. 서술용.
확실한 짝 = linker_v3/t3.py 의 cert 연결(학습 정답의 23시→0시 이어짐이 뚜렷한 1:1 짝: c1 ≤ 1, c2 − c1 ≥ 1, 외기 이어짐 oj1 ≤ 1.2, 중복 제거).
그 가운데 앞 기록(a1)과 뒤 기록(b)의 온실 ID가 다른 비율을 센다. 구간(1·2차)·방향별로도 나눈다."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: F401
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", "linker_v3", "truth.pkl"))
C = T[T.cert]
x = C.fa != C.fb
print(f"확실한 짝 {len(C)}개, 다른 ID {int(x.sum())}개 = {x.mean():.3f}")
print("뒤 기록 구간별:", {f"{s}차": f"{int(x[C.segb == s].sum())}/{int((C.segb == s).sum())} = {x[C.segb == s].mean():.3f}" for s in (1, 2)})
print("방향별(앞 ID → 뒤 ID):")
print(pd.crosstab(C.fa, C.fb).to_string())
