"""Stage a local, source-complete candidate package from the audited v7 ZIP."""
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
ORIGINAL_ZIP = SOURCE / 'research/submissions/팜모니_정형데이터_재현패키지_07.zip'
WEIGHTS = Path('C:/Users/aozks/AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt')
CANDIDATE = ROOT / 'local/ec_tabpfn_local_candidate/20260928_032419/candidate_temp06_ec_tabpfn_v2_cpu.csv'
NAME = 'candidate_temp06_ec_tabpfn_v2_cpu.csv'
ROOT_NAME = 'farmmoni_tabpfn_ec_candidate'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert ORIGINAL_ZIP.is_file() and WEIGHTS.is_file() and CANDIDATE.is_file()
    assert sha(WEIGHTS) == '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    out = ROOT / 'local/ec_tabpfn_reproduction' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    pkg = out / 'stage' / ROOT_NAME
    pkg.mkdir(parents=True)
    with zipfile.ZipFile(ORIGINAL_ZIP) as old:
        for item in old.infolist():
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            assert parts[0] == '팜모니_정형데이터_재현패키지'
            assert len(parts) >= 2 and '..' not in parts and not PurePosixPath(item.filename).is_absolute()
            relative = Path(*parts[1:])
            if relative == Path('README.md'):
                relative = Path('README_previous_07.md')
            elif relative == Path('submission_07.csv'):
                relative = Path('reference/submission_07.csv')
            elif relative == Path('requirements.txt'):
                continue
            target = pkg / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old.read(item))

    shutil.copy2(HERE.with_name('README.md'), pkg / 'README.md')
    shutil.copy2(HERE.with_name('reproduce_candidate.py'), pkg / 'code/reproduce_candidate.py')
    shutil.copy2(HERE.parents[1] / 'ec_tabpfn_local_candidate/PROTOCOL.md', pkg / 'config/candidate_protocol.md')
    shutil.copy2(CANDIDATE, pkg / 'reference' / NAME)
    model_dir = pkg / 'model'
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(WEIGHTS, model_dir / WEIGHTS.name)
    shutil.copy2(HERE.parent / 'model/LICENSE.txt', model_dir / 'LICENSE.txt')
    (pkg / 'requirements.txt').write_text(
        'numpy==2.5.3\npandas==3.0.6\nscipy==1.18.1\nscikit-learn==1.9.1\n'
        'lightgbm==4.7.0\ntorch==2.14.0\ntabpfn==9.0.0\n', encoding='utf-8')

    archive = out / 'farmmoni_tabpfn_ec_candidate_reproduction.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(pkg.rglob('*')):
            if file.is_file():
                z.write(file, arcname=Path(ROOT_NAME) / file.relative_to(pkg))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        paths = z.namelist()
    manifest = dict(source_zip_sha256=sha(ORIGINAL_ZIP), candidate_sha256=sha(CANDIDATE),
                    model_sha256=sha(WEIGHTS), model_license_sha256=sha(model_dir / 'LICENSE.txt'),
                    reproduce_code_sha256=sha(HERE.with_name('reproduce_candidate.py')),
                    archive_sha256=sha(archive), archive_files=len(paths),
                    excludes_raw_competition_data=True, platform_submitted=False,
                    archive=str(archive), stage=str(pkg))
    (out / 'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
