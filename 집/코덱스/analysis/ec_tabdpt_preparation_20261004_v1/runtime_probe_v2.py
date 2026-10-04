"""TabDPT v2 import-only probe: 엄격한 metadata/path/5-pin/commit/DLL/API 확인.
checkpoint constructor, weights, 데이터, fit, predict, 신규 설치·다운로드 호출 없음.
"""
from __future__ import annotations
import argparse
import base64
from email.parser import Parser
import hashlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parents[3]
SITE = ROOT / "집" / "코덱스" / "local" / "tabdpt130_cpu_v1" / "site"
CORE = ROOT / ".analysis-tools" / "python"
EXTRA = ROOT / ".analysis-tools" / "extra"
COMMIT = "97e5494431e9527c7edb31cb4dcfc5f00b232fdf"
EXACT = {"tabdpt": "1.3.0", "faiss-cpu": "1.12.0",
         "omegaconf": "2.3.0", "antlr4-python3-runtime": "4.9.3",
         "huggingface-hub": "0.36.0"}
REQUIRED = {"numpy": ">=1.25,<3", "scikit-learn": ">=1.4,<2",
            "scipy": ">=1.9,<2", "torch": ">=2.6,<3",
            "safetensors": ">=0.5.3,<1", "tqdm": ">=4.38,<5",
            "huggingface-hub": ">=0.33.2,<2",
            "faiss-cpu": ">=1.11,<1.13", "omegaconf": ">=2.1.1,<3"}
MODULES = {"tabdpt": ("tabdpt", SITE), "faiss-cpu": ("faiss", SITE),
           "omegaconf": ("omegaconf", SITE),
           "antlr4-python3-runtime": ("antlr4", SITE),
           "huggingface-hub": ("huggingface_hub", SITE),
           "numpy": ("numpy", CORE), "scikit-learn": ("sklearn", CORE),
           "scipy": ("scipy", CORE), "torch": ("torch", EXTRA),
           "safetensors": ("safetensors", EXTRA), "tqdm": ("tqdm", EXTRA)}
DLL_HANDLES = []


def normal(name):
    return name.lower().replace("_", "-").replace(".", "-")


def strict_resolve(name, roots):
    """첫 경로의 정상 METADATA를 직접 읽는다. None/권한 오류를 숨기지 않는다."""
    wanted = normal(name)
    for root in roots:
        if not root.is_dir():
            continue
        matches = [p for p in root.glob("*.dist-info")
                   if normal(p.name.removesuffix(".dist-info").rsplit("-", 1)[0]) == wanted]
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError(f"{name}: 같은 경로에서 dist-info가 여러 개입니다: {matches}")
        p = matches[0]
        text = (p / "METADATA").read_text(encoding="utf-8")
        message = Parser().parsestr(text)
        if normal(message.get("Name", "")) != wanted or not message.get("Version"):
            raise RuntimeError(f"{name}: METADATA Name/Version 누락: {p}")
        dist = md.PathDistribution(p)
        if dist.version != message["Version"]:
            raise RuntimeError(f"{name}: parser 간 version 불일치: {p}")
        return dist, {"name": message["Name"], "version": message["Version"],
                      "dist_info": str(p), "metadata_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "root": str(root)}
    raise md.PackageNotFoundError(name)


def probe(site):
    site = site.resolve(strict=True)
    if site != SITE.resolve(strict=True):
        raise RuntimeError("고정 overlay site 경로 불일치")
    if any(n in sys.modules for n in ("torch", "numpy", "tabdpt", "faiss", "huggingface_hub")):
        raise RuntimeError("깨끗한 Python worker가 필요합니다")
    sys.path.insert(0, str(ROOT / "집" / "클로드" / "research"))
    import env  # 첫 project import
    import env_extra
    sys.path.insert(0, str(site))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dll_dirs = {ROOT / ".analysis-tools" / "msvc", EXTRA / "torch" / "lib", site / "faiss"}
    for root in (CORE, EXTRA, site):
        dll_dirs.update(root.glob("*.libs"))
        dll_dirs.update(root.glob("*/.libs"))
    live_dll_dirs = []
    if hasattr(os, "add_dll_directory"):
        for directory in sorted(dll_dirs):
            if directory.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(directory)))
                live_dll_dirs.append(str(directory))

    from packaging.version import Version
    from packaging.specifiers import SpecifierSet
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    roots = list(dict.fromkeys([site, CORE, EXTRA] +
                               [Path(p) for p in sys.path if p]))
    resolved, records = {}, {}
    for name in list(EXACT) + list(REQUIRED):
        if name not in resolved:
            dist, record = strict_resolve(name, roots)
            resolved[name], records[name] = dist, record
    for name, pin in EXACT.items():
        if records[name]["root"] != str(site) or records[name]["version"] != pin:
            raise RuntimeError(f"{name}: 고정 overlay exact pin 불일치: {records[name]}")
    for name, spec in REQUIRED.items():
        if Version(records[name]["version"]) not in SpecifierSet(spec):
            raise RuntimeError(f"{name}: 기존 version 문턱 {spec} 불일치")
    direct = json.loads(resolved["tabdpt"].read_text("direct_url.json"))
    if direct.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise RuntimeError("TabDPT 전체 VCS commit 불일치")

    # 설치된 실제 모듈 파일이 설치 RECORD와 일치하는지 추가 검산.
    record_checks = []
    for item in resolved["tabdpt"].files or []:
        if str(item).startswith("tabdpt/") and str(item).endswith(".py"):
            path = resolved["tabdpt"].locate_file(item)
            value = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
            if item.hash is None or item.hash.mode != "sha256" or value != item.hash.value:
                raise RuntimeError(f"TabDPT source RECORD hash 불일치: {item}")
            record_checks.append(str(item))
    if not record_checks:
        raise RuntimeError("TabDPT source RECORD 검사 대상이 없습니다")

    # 추가 설치 없이 활성 전이 의존성의 현재 metadata 범위를 확인.
    dependency_checks = []
    marker_env = default_environment()
    marker_env["extra"] = ""
    queue = list(EXACT) + list(REQUIRED)
    checked = set()
    while queue:
        name = queue.pop(0)
        key = normal(name)
        if key in checked:
            continue
        checked.add(key)
        if name not in resolved:
            resolved[name], records[name] = strict_resolve(name, roots)
        for line in resolved[name].requires or []:
            req = Requirement(line)
            if req.marker and not req.marker.evaluate(marker_env):
                continue
            if req.name not in resolved:
                resolved[req.name], records[req.name] = strict_resolve(req.name, roots)
            version = records[req.name]["version"]
            if req.specifier and Version(version) not in req.specifier:
                raise RuntimeError(f"활성 의존성 불일치 {name}: {req}, 실제 {version}")
            dependency_checks.append({"parent": name, "requirement": str(req),
                                      "selected_version": version,
                                      "dist_info": records[req.name]["dist_info"]})
            queue.append(req.name)

    paths = {}
    for distribution, (module_name, root) in MODULES.items():
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to((root / module_name).resolve()):
            raise RuntimeError(f"{module_name}: 실제 module 경로 불일치: {path}")
        value = getattr(module, "__version__", None)
        if value is not None:
            v = Version(str(value))
            m = Version(records[distribution]["version"])
            if v.base_version != m.base_version:
                raise RuntimeError(f"{module_name}: module version과 metadata version 불일치")
        paths[distribution] = {"module": module_name, "module_path": str(path),
                               "module_version": None if value is None else str(value),
                               "metadata_version": records[distribution]["version"]}
    from tabdpt.regressor import TabDPTRegressor
    constructor, predict = inspect.signature(TabDPTRegressor.__init__), inspect.signature(TabDPTRegressor.predict)
    for key in ("normalizer", "missing_indicators", "clip_sigma", "feature_reduction",
                "context_reduction", "faiss_metric", "device", "use_flash", "compile",
                "model_weight_path", "verbose"):
        if key not in constructor.parameters:
            raise RuntimeError(f"constructor API missing {key}")
    for key in ("n_ensembles", "context_size", "batch_size", "seed", "output_type"):
        if key not in predict.parameters:
            raise RuntimeError(f"predict API missing {key}")
    return {"status": "PASS_IMPORT_ONLY_FIT_PREDICT_UNTESTED",
            "python": sys.version, "executable": sys.executable, "site": str(site),
            "exact_five_pins": EXACT, "metadata": records, "module_paths": paths,
            "tabdpt_direct_url": direct, "tabdpt_source_record_checks": record_checks,
            "active_dependency_checks": dependency_checks,
            "dll_dirs_with_live_handles": live_dll_dirs,
            "constructor_signature": str(constructor), "predict_signature": str(predict),
            "new_installs": 0, "weight_reads": 0, "new_weight_downloads": 0,
            "data_reads": 0, "fit": 0, "predict": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=SITE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = probe(args.runtime_root)
    except Exception as error:
        result = {"status": "FAIL_IMPORT_ONLY", "error_type": type(error).__name__,
                  "error": str(error), "python": sys.version, "executable": sys.executable,
                  "new_installs": 0, "new_weight_downloads": 0, "data_reads": 0,
                  "weight_reads": 0, "fit": 0, "predict": 0}
    if args.output:
        target = args.output.resolve()
        if not target.is_relative_to(Path(__file__).resolve().parent) or target.exists():
            raise RuntimeError("새 own 준비 폴더 파일에만 결과를 저장합니다")
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"].startswith("PASS") else 1)

