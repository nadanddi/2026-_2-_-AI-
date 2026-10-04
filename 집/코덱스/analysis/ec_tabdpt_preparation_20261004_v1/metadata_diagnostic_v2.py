"""stdlib 배포 metadata만 읽는 진단. env/타사 모듈/데이터/weight import 0."""
from pathlib import Path
import importlib.metadata as md
import json
import sys
import platform

ROOT = Path(__file__).resolve().parent.parents[3]
SITE = ROOT / "집" / "코덱스" / "local" / "tabdpt130_cpu_v1" / "site"
PATHS = [SITE, ROOT / ".analysis-tools" / "python",
         ROOT / ".analysis-tools" / "extra"]
NAMES = {"tabdpt","faiss-cpu","omegaconf","antlr4-python3-runtime","huggingface-hub",
         "numpy","scikit-learn","scipy","torch","safetensors","tqdm",
         "packaging","pyyaml","filelock","fsspec","requests","typing-extensions","hf-xet"}

def norm(value):
    return value.lower().replace("_","-").replace(".","-")

records=[]
for root in PATHS:
    for d in md.distributions(path=[str(root)]):
        p=Path(d._path)
        fallback=p.name.removesuffix(".dist-info").rsplit("-",1)[0]
        name=d.metadata.get("Name")
        if norm(name or fallback) in NAMES:
            records.append({"search_root":str(root),"dist_info":str(p),
                            "metadata_exists":(p/"METADATA").is_file(),
                            "metadata_size":(p/"METADATA").stat().st_size if (p/"METADATA").is_file() else None,
                            "name":name,"fallback_name":fallback,"version":d.version,
                            "files":sorted(x.name for x in p.iterdir()),
                            "requirements":d.requires or []})
initial_paths=list(sys.path)
sys.path[:0]=[str(x) for x in PATHS]
first=[]
for name in sorted(NAMES):
    try:
        d=md.distribution(name)
        first.append({"request":name,"name":d.metadata.get("Name"),"version":d.version,
                      "dist_info":str(d._path)})
    except md.PackageNotFoundError:
        first.append({"request":name,"status":"NOT_FOUND"})
report=json.loads((ROOT/"집"/"코덱스"/"analysis"/"ec_tabdpt_20261004_v1"/"install_report_v1.json").read_text(encoding="utf-8"))
print(json.dumps({"status":"READ_ONLY_METADATA","python":sys.version,
                  "executable":sys.executable,"initial_sys_path":initial_paths,
                  "searched_paths":[str(x) for x in PATHS],
                  "distribution_records":records,"default_resolution":first,
                  "parent_installed":[{"name":x["metadata"]["name"],"version":x["metadata"]["version"],
                                       "download_info":x["download_info"]}
                                      for x in report["install"]],
                  "third_party_imports":0,"weight_reads":0,"data_reads":0},
                 ensure_ascii=False,indent=2))

