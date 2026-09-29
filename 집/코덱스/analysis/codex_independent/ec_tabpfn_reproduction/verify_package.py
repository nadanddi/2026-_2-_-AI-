"""Extract the package into a fresh local folder and rerun it offline."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
BUILD = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
         else ROOT / 'local/ec_tabpfn_reproduction/20260928_033057')
assert BUILD.is_dir() and ROOT / 'local/ec_tabpfn_reproduction' in BUILD.parents
ZIP = BUILD / 'farmmoni_tabpfn_ec_candidate_reproduction.zip'
TEST = BUILD / 'fresh_extraction'
PKG = TEST / 'farmmoni_tabpfn_ec_candidate'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert ZIP.is_file() and not TEST.exists()
    TEST.mkdir(parents=True)
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            assert parts and parts[0] == PKG.name and '..' not in parts
            assert not PurePosixPath(info.filename).is_absolute()
            assert not any(part in ('train_X.csv', 'train_y.csv', 'test_X.csv') for part in parts)
        z.extractall(TEST)
    data_dir = PKG / 'data'
    for name in ('train_X.csv', 'train_y.csv', 'test_X.csv'):
        shutil.copy2(SOURCE / '온라인대회자료/정형데이터/참가자_배포' / name, data_dir / name)
    dependencies = SOURCE / '.analysis-tools'
    child_env = os.environ.copy()
    child_env['AGRI_PACKAGE_DEPS_ROOT'] = str(dependencies)
    child_env['PYTHONPATH'] = os.pathsep.join((str(dependencies / 'python'), str(dependencies / 'extra')))
    child_env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, '-u', str(PKG / 'code/reproduce_candidate.py')]
    with (BUILD / 'reproduction.log').open('w', encoding='utf-8') as logfile:
        proc = subprocess.Popen(command, cwd=PKG / 'code', env=child_env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            logfile.write(line)
            logfile.flush()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f'Fresh package reproduction exited {code}')
    result = PKG / 'output/candidate_temp06_ec_tabpfn_v2_cpu.csv'
    expected = PKG / 'reference/candidate_temp06_ec_tabpfn_v2_cpu.csv'
    assert result.is_file() and expected.is_file()
    assert result.read_bytes() == expected.read_bytes(), 'Fresh output bytes differ from candidate'
    manifest = json.loads((PKG / 'output/manifest_tabpfn_v2_cpu.json').read_text(encoding='utf-8'))
    assert manifest['output_sha256'] == sha(result)
    report = dict(status='PASS', output_sha256=sha(result),
                  expected_sha256=sha(expected), output_bytes_match=True,
                  package_sha256=sha(ZIP), offline=True, exit_code=code)
    (BUILD / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
