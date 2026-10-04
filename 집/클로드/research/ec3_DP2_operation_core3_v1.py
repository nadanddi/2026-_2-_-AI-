# -*- coding: utf-8 -*-
"""EC stage-3 DP2: DP1 reduced to 3 operation-sequence features directly tied to the
high-EC condition (fixed before running; 2026-10-04 집 클로드).
POST-HOC variant of DP1 (C6.256: DIAG10 -1.3~-1.7%, A -0.6~-1.4%, B +0.2~+0.8%,
P .047-.107): keep seal_run, co2_hours, heat_run only (sealed / CO2 / heating, the
HC0 high-EC signals), drop the other 6 to cut noise.  Because it is chosen after
seeing DP1, the family is k = 2 -> DIAG10 P(worse) threshold .0125.
Judge (EC rule 2026-10-04): every seed x DIAG10/A/B better and DIAG10 P(worse) <
.0125; guard: EL1 or DIAG10 pass-2 rows worse by >= 2 % -> HOLD; EXT reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DP2_operation_core3_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dp1.NEW = ["seal_run", "co2_hours", "heat_run"]
dp1.CK = dp1.db1.CK = os.path.join(env.LOCAL, "dp2_ckpt")
dp1.TAG = "dp2"
P_THR = 0.0125
_orig = dp1.db1.judge_report
def judge_report(TAG, NAME):
    import builtins
    src = open(dp1.db1.__file__, encoding="utf-8").read()
    # same report as DB1/DP1 but with the k=2 threshold
    ns = {}
    code = src.split("def judge_report")[1].split("\n\nif __name__")[0]
    exec("def judge_report" + code.replace("all(p < 0.025 for p in ps)", "all(p < %s for p in ps)" % P_THR), dp1.db1.__dict__)
    dp1.db1.judge_report(TAG, NAME)
dp1.judge_report_k2 = judge_report
if __name__ == "__main__":
    # run DP1 main with the overridden globals, then the k=2 report
    import types
    main_src = open(dp1.__file__, encoding="utf-8").read().split("def main():")[1].split("\n\nif __name__")[0]
    main_src = main_src.replace('db1.judge_report(TAG, "DP1")', 'judge_report_k2(TAG, "DP2")')
    exec("def main():" + main_src, dp1.__dict__)
    dp1.main()
