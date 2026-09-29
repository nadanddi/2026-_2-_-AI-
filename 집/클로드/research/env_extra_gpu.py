# -*- coding: utf-8 -*-
"""Like env_extra, but the CUDA build of torch (.analysis-tools/extra_gpu)
takes precedence over the CPU build in .analysis-tools/extra.  Experiments
only (speed); anything shipped must reproduce on the CPU environment.

Import AFTER `env`, INSTEAD of env_extra.
"""
import os
import sys

import env

TOOLS_ROOT = os.path.dirname(env.TOOLS)
GPU = os.path.join(TOOLS_ROOT, "extra_gpu")
EXTRA = os.path.join(TOOLS_ROOT, "extra")
MSVC = os.path.join(TOOLS_ROOT, "msvc")

if hasattr(os, "add_dll_directory"):
    for d in (MSVC, os.path.join(GPU, "torch", "lib")):
        if os.path.isdir(d):
            os.add_dll_directory(d)
for p in (GPU, EXTRA):
    if p not in sys.path:
        sys.path.append(p)
