# -*- coding: utf-8 -*-
"""Temperature half of tabpfn_cpu_verify_v1.py (the EC half passed, catalog
6.59; the run was stopped by the user before the temperature half finished).

Run:  cd research && PYTHONPATH="" <python> -u tabpfn_cpu_verify_temp_v1.py
"""
import env  # noqa: F401
import env_extra  # noqa: F401
from tabpfn_cpu_verify_v1 import temperature

if __name__ == "__main__":
    temperature()
