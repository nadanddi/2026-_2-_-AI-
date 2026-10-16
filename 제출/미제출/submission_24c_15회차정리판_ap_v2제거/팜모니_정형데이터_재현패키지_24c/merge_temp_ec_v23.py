# -*- coding: utf-8 -*-
"""submission_23 = sub_temp from temp_candidate_v23 (round-13 temperature + TNA8 extra anchors, hypothesis test)
+ sub_ec from EC/output_23/ec_b5.csv (round-12 EC + B5 same-date high-EC exclusion, hypothesis test),
joined by row_id.  Checks: unique ids, same id set, 1440 rows, finite values,
temperature-file row order (= sample_submission).  UTF-8 without BOM, 6 decimals
like the earlier submissions.

Run:  python merge_temp_ec_v23.py <temp_candidate_v23.csv> <EC b5 csv> <output csv>
"""
import os
import sys

import numpy as np
import pandas as pd

tp, ep, op = sys.argv[1:4]
assert not os.path.exists(op), "refusing to overwrite %s" % op
t = pd.read_csv(tp, encoding="utf-8-sig")
e = pd.read_csv(ep, encoding="utf-8-sig")[["row_id", "sub_ec"]]
assert {"row_id", "sub_temp"} <= set(t.columns)
assert t.row_id.is_unique and e.row_id.is_unique and len(t) == len(e) == 1440 and set(t.row_id) == set(e.row_id)
q = t[["row_id", "sub_temp"]].merge(e, on="row_id", how="left", validate="one_to_one", sort=False)
assert q.row_id.tolist() == t.row_id.tolist() and np.isfinite(q[["sub_temp", "sub_ec"]].to_numpy(float)).all()
q.to_csv(op, index=False, encoding="utf-8", float_format="%.6f")
print("wrote", op)
