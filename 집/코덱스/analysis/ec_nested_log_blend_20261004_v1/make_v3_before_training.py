"""Add runtime version pinning before any family19 model fit."""
from pathlib import Path
H = Path(__file__).resolve().parent
src = (H/'run_v2.py').read_text(encoding='utf-8')
dest = H/'run_v3.py'
assert not dest.exists()
src = src.replace("deps = dict(core=S.sha(Path(core.__file__)),", '''import platform, sklearn, lightgbm
    runtime = dict(python=platform.python_version(), numpy=np.__version__,
                   pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__)
    expected_runtime = dict(python='3.12.14', numpy='2.5.3', pandas='3.0.1',
                            sklearn='1.9.1', lightgbm='4.7.0')
    assert runtime == expected_runtime, ('runtime changed', runtime)
    deps = dict(core=S.sha(Path(core.__file__)),''')
src = src.replace("assert original_meta['provenance']['shared']['input_sha256']['train_X.csv'] == input_sha", '''assert original_meta['provenance']['shared']['input_sha256']['train_X.csv'] == input_sha
            assert all(original_meta['provenance']['environment'][name] == value for name,value in runtime.items())''')
src = src.replace("check.update(dependency_sha256=deps,", "check.update(runtime=runtime, dependency_sha256=deps,")
src = src.replace("H / 'preparation_v2.json'", "H / 'preparation_v3.json'")
src = src.replace("dependency_sha256=deps, train_input_sha256=input_sha, public_cache_sha256=public_cache_sha,", "runtime=runtime, dependency_sha256=deps, train_input_sha256=input_sha, public_cache_sha256=public_cache_sha,")
compile(src,str(dest),'exec')
dest.write_text(src,encoding='utf-8')
v = (H/'verify_full_v2.py').read_text(encoding='utf-8').replace("eh/'run_v2.py'", "eh/'run_v3.py'")
vd = H/'verify_full_v3.py'; assert not vd.exists()
compile(v,str(vd),'exec'); vd.write_text(v,encoding='utf-8')
print('v3 source/runtime guard created; no fit')
