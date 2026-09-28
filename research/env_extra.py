# -*- coding: utf-8 -*-
"""Optional packages (GPBoost, TabPFN/torch) for experiments only.

Import AFTER `env`.  The extra packages live in .analysis-tools/extra and are
APPENDED to sys.path, so the pinned core versions in .analysis-tools/python
(numpy 2.5.3, pandas 3.0.1, scikit-learn 1.9.1, lightgbm 4.7.0) keep
precedence.  The MSVC runtime (msvcp140_2.dll etc., missing on this machine)
comes from the `msvc-runtime` wheel in .analysis-tools/msvc and is registered
as a DLL directory.  Nothing here is used by the submission generators.
"""
import os
import sys

import env

TOOLS_ROOT = os.path.dirname(env.TOOLS)
EXTRA = os.path.join(TOOLS_ROOT, "extra")
MSVC = os.path.join(TOOLS_ROOT, "msvc")

if hasattr(os, "add_dll_directory"):
    for d in (MSVC, os.path.join(EXTRA, "torch", "lib")):
        if os.path.isdir(d):
            os.add_dll_directory(d)
if EXTRA not in sys.path:
    sys.path.append(EXTRA)
