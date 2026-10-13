# -*- coding: utf-8 -*-
"""SV1B: SV1 저장 결과(sv1_draws_v1.csv, sv1_situations_v1.csv)로 닮음 확인 표만 다시 출력(SV1 마지막 출력 줄 인코딩 오류 — 계산은 저장 완료)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "sv1_structure_validator_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "sv1_structure_validator_v1.py")
exec(src[:src.index("D, S = [], []")])
S = pd.read_csv(os.path.join(HERE, "sv1_situations_v1.csv"))
E = pd.DataFrame([dict(rec=rid(i), **situ(i, LAB)) for i in range(N) if not TRAIN[i]])
pd.set_option("display.width", 200); cols = ["date_found", "S1", "S2", "S3", "twin"]
T = pd.concat([E[cols].mean().rename("EVAL (60)"), S[S.pass2][cols].mean().rename(f"pseudo pass-2 val (n/draw {S[S.pass2].groupby('draw').size().mean():.0f})"),
               S[~S.pass2][cols].mean().rename(f"pseudo pass-1 val (n/draw {S[~S.pass2].groupby('draw').size().mean():.0f})")], axis=1)
print("situation shares (S2 for EVAL unknown - no truth):"); print(T.round(3).to_string())
print("\npseudo val high-EC share: pass-2", round(float(S[S.pass2].hi.mean()), 3), "| pass-1", round(float(S[~S.pass2].hi.mean()), 3))
print("pseudo pass-2 val: unique records", S[S.pass2].rec.nunique(), "| high", int(S[S.pass2].drop_duplicates('rec').hi.sum()))
print(pd.crosstab([S[S.pass2].S1, S[S.pass2].S2], S[S.pass2].hi).to_string())
