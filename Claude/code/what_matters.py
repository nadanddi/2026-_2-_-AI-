# -*- coding: utf-8 -*-
"""What actually drives the score: greenhouse physics, or position in time?

Each target is fitted on three restricted views, scored on the test-geometry CV:
  TIME   : day / hour / farm only -- no sensor reading at all
  SENSOR : every sensor-derived feature, with day and farm removed
  BOTH   : the full submitted view
A fourth row gives the constant-per-greenhouse baseline for scale.
One seed per model (this is a decomposition probe, not a model choice).
"""
import numpy as np
import lightgbm as lgb

from common import rmse, TARGET_FARMS, USABLE
from model_v2 import get_panel
from make_submission import T_HUB, E_HUB, V5_ET, DOMAIN_MARKS
from geometry_cv import geometry_folds, score, lgb_fp, et_fp
from common import split_mask
import features_v2 as F2

TIME_COLS = ["day", "hour", "hr_sin", "hr_cos", "farm_id"]


def baseline(lab, folds, target):
    """Per-greenhouse mean of the training part of each fold."""
    lates = []
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        m = tr.groupby("farm")[target].mean()
        lates.append(rmse(va.farm.map(m).values, va[target].values))
    return float(np.mean(lates))


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    folds = geometry_folds()

    v_temp = F2.view(panel, "sub_temp")
    v_ec_cur = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
    v_ec_v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]

    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)

    def nt(cols):   # sensor-only: strip the calendar
        return [c for c in cols if c not in TIME_COLS + ["midnight", "day_par"]]

    print("=== sub_temp  (기준선 = 온실별 평균) ===")
    print("  %-34s %.4f" % ("기준선 (온실 평균만)", baseline(lab_t, folds, "sub_temp")))
    print("  %-34s %.4f" % ("TIME 만 (day/hour/farm)",
                            score(lab_t, folds, "sub_temp", lgb_fp(TIME_COLS, T_HUB), )[0]))
    print("  %-34s %.4f" % ("현재 실내온도 1개만",
                            score(lab_t, folds, "sub_temp", lgb_fp(["in_temp"], T_HUB))[0]))
    print("  %-34s %.4f" % ("SENSOR 만 (달력 제거, %d개)" % len(nt(v_temp)),
                            score(lab_t, folds, "sub_temp", lgb_fp(nt(v_temp), T_HUB))[0]))
    print("  %-34s %.4f" % ("BOTH = 제출 구성 (%d개)" % len(v_temp),
                            score(lab_t, folds, "sub_temp", lgb_fp(v_temp, T_HUB))[0]))

    print("\n=== sub_ec  (기준선 = 온실별 평균) ===")
    print("  %-34s %.4f" % ("기준선 (온실 평균만)", baseline(lab_e, folds, "sub_ec")))
    print("  %-34s %.4f" % ("TIME 만 (day/hour/farm)",
                            score(lab_e, folds, "sub_ec", et_fp(TIME_COLS, V5_ET))[0]))
    print("  %-34s %.4f" % ("day + farm 2개만",
                            score(lab_e, folds, "sub_ec", et_fp(["day", "farm_id"], V5_ET))[0]))
    print("  %-34s %.4f" % ("SENSOR 만 (달력 제거, %d개)" % len(nt(v_ec_v5)),
                            score(lab_e, folds, "sub_ec", et_fp(nt(v_ec_v5), V5_ET))[0]))
    print("  %-34s %.4f" % ("BOTH = 2회차 v5 (%d개)" % len(v_ec_v5),
                            score(lab_e, folds, "sub_ec", et_fp(v_ec_v5, V5_ET))[0]))
    print("  %-34s %.4f" % ("BOTH = 1회차 이력 68개",
                            score(lab_e, folds, "sub_ec", et_fp(v_ec_cur, V5_ET))[0]))


if __name__ == "__main__":
    main()
