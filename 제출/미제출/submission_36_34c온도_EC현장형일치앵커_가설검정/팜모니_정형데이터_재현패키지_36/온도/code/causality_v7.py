# -*- coding: utf-8 -*-
"""Rule checks for submission_07: causality_v6.py's checks on the columns
manifest_07 feeds a model (the same 150 columns as submission_06; the EC
ExtraTrees member uses `day` again, which is already among them).

Run:  cd research && PYTHONPATH="" <python> -u causality_v7.py   (ALL PASS)
"""
import os
import sys

import env  # noqa: F401
import causality_v6 as C


def man_path():
    for p in (os.path.join(C.HERE, "..", "config", "manifest_07.json"),
              os.path.join(C.HERE, "submissions", "manifest_07.json")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("manifest_07.json")


C.man_path = man_path

if __name__ == "__main__":
    sys.exit(C.main())
