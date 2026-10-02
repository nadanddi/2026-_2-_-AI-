"""Resume all five registered candidates after current workers have stopped.

Existing artifacts are preserved. No submission or locked-label evaluation.
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '집/코덱스/local/statistical_experiments_20261003_v1'
STAGES = [
    ('prepare_v2.py', OUT/'plan.json'),
    ('gpu.py', OUT/'gpu_done.json'),
    ('analyze.py', HERE/'result.json'),
    ('verify.py', HERE/'verification.json'),
    ('gap_audit.py', HERE/'gap_audit.json'),
    ('verify_gap_v2.py', HERE/'gap_verification.json'),
    ('replay_cpu.py', HERE/'cpu_replay.json'),
    ('conditional_audit.py', HERE/'conditional_audit.json'),
    ('verify_conditional.py', HERE/'conditional_verification.json'),
    ('verify_bootstrap.py', HERE/'bootstrap_verification.json'),
    ('extra_v2.py', HERE/'extra_result.json'),
    ('verify_extra.py', HERE/'extra_verification.json'),
    ('weather_audit.py', HERE/'weather_audit.json'),
    ('deployment_gap_audit.py', HERE/'deployment_gap_audit.json'),
    ('environment.py', HERE/'environment.json'),
    ('oracle_limits.py', HERE/'oracle_limits.json'),
    ('verify_oracle.py', HERE/'oracle_verification.json'),
    ('report_v2.py', HERE/'종합_결과보고서_v2.md'),
]

if __name__ == '__main__':
    for script, artifact in STAGES:
        if artifact.exists():
            print(f'PRESERVED {artifact.name}', flush=True)
            continue
        subprocess.run([sys.executable, '-u', str(HERE/script)], check=True)
