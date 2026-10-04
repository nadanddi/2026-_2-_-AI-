"""미실행 환경 probe 초안. 기본 실행은 명세만 출력하며 --check-imports는 향후 담당자용.
check-imports도 checkpoint constructor, fit, predict, 데이터 로더를 호출하지 않는다.
새 환경 설치와 공개 접근/라이선스 검토 후 별도 worker에서만 사용한다.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parents[3]
DLL_HANDLES = []
PROPOSED_SITE = ROOT / "집" / "코덱스" / "local" / "tabdpt130_cpu_v1" / "site"


def import_probe(runtime_root: Path) -> dict:
    """향후 실행: 데이터/weight 없는 의존성 import 및 API 서명 확인."""
    runtime_root = runtime_root.resolve(strict=True)
    if not runtime_root.is_relative_to((ROOT / "집" / "코덱스" / "local").resolve()):
        raise ValueError("독립 runtime 경로가 집/코덱스/local 아래여야 합니다.")
    if any(name in sys.modules for name in
           ("tabdpt", "faiss", "omegaconf", "huggingface_hub", "torch", "safetensors")):
        raise RuntimeError("새 Python worker가 필요합니다.")

    sys.path.insert(0, str(ROOT / "집" / "클로드" / "research"))
    import env  # 첫 project import
    import env_extra
    sys.path.insert(0, str(runtime_root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 공용 bootstrap은 변경하지 않는다. 이 worker에서는 핸들을 유지한다.
    candidates = {
        ROOT / ".analysis-tools" / "msvc",
        ROOT / ".analysis-tools" / "extra" / "torch" / "lib",
        runtime_root / "faiss",
    }
    for package_root in (ROOT / ".analysis-tools" / "python",
                         ROOT / ".analysis-tools" / "extra", runtime_root):
        candidates.update(package_root.glob("*.libs"))
        candidates.update(package_root.glob("*/.libs"))
    dll_dirs = []
    if hasattr(os, "add_dll_directory"):
        for directory in sorted(candidates):
            if directory.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(directory)))
                dll_dirs.append(str(directory))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import adapter_draft_v1 as adapter
    versions = adapter.verify_package_pin(execution_authorized=True)
    modules = {name: importlib.import_module(name) for name in
               ("numpy", "scipy", "sklearn", "torch", "safetensors",
                "faiss", "omegaconf", "huggingface_hub")}
    from tabdpt.regressor import TabDPTRegressor
    constructor = inspect.signature(TabDPTRegressor.__init__)
    predict = inspect.signature(TabDPTRegressor.predict)
    if any(key not in constructor.parameters for key in adapter.CONSTRUCTOR):
        raise RuntimeError("constructor API 불일치")
    if any(key not in predict.parameters for key in adapter.PREDICT):
        raise RuntimeError("predict API 불일치")
    return {
        "status": "IMPORT_ONLY_FIT_AND_PREDICT_UNTESTED",
        "versions": versions,
        "module_paths": {name: str(module.__file__) for name, module in modules.items()},
        "runtime_root": str(runtime_root),
        "dll_dirs_with_live_handles": dll_dirs,
        "constructor_signature": str(constructor),
        "predict_signature": str(predict),
        "data_reads": 0,
        "weight_reads": 0,
        "weight_downloads": 0,
        "fits": 0,
        "predictions": 0,
        "source_pin_check": "direct_url.json full VCS commit",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=PROPOSED_SITE)
    parser.add_argument("--check-imports", action="store_true",
                        help="향후 설치·검토 완료 후 별도 worker의 import 검사만")
    args = parser.parse_args()
    result = import_probe(args.runtime_root) if args.check_imports else {
        "status": "DRAFT_NO_IMPORTS_EXECUTED",
        "proposed_runtime_root": str(args.runtime_root),
        "imports": 0, "data_reads": 0, "weight_reads": 0, "fits": 0,
        "note": "dependency lock and parent review pending",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

