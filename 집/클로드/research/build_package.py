# -*- coding: utf-8 -*-
"""Build the reproduction ZIP required by the problem statement, section 8.2.

The platform takes two uploads: the answer CSV (scored) and a ZIP holding the
code/reproduction package (kept for review, never executed automatically).
Section 8.2 asks for four things, and each maps to a part of the ZIP:

  코드           -> code/  (preprocessing, training, inference, audits)
  모델과 설정     -> config/manifest_03.json + the deterministic pipeline
  실행 안내       -> README.md and requirements.txt
  자료 사용 내역   -> README.md section 4

The packaged env.py is a portable rewrite of the research one: it resolves the
data directory to <package>/data (or $AGRI_DATA) and does not depend on this
machine's bundled-DLL workaround.

Run:  cd research && PYTHONPATH="" <python> build_package.py
"""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE_SRC = os.path.join(ROOT, "Claude", "code")
STAGE = os.path.join(HERE, "local", "package")
TEAM = "팜모니_정형데이터_재현패키지"

FROM_RESEARCH = ["harness.py", "feat_lib.py", "feat_new.py", "feat_temp74.py",
                 "make_submission_v3.py", "causality_new.py",
                 "causality_shrink.py"]
FROM_CLAUDE = ["common.py", "features_v2.py", "causality_test.py"]

PORTABLE_ENV = '''# -*- coding: utf-8 -*-
"""Bootstrap for the reproduction package.

Resolves the competition CSVs and the output directory relative to this
package, so the reviewer only has to drop the three files into data/.

  <package>/data/train_X.csv, train_y.csv, test_X.csv   (put them here)
  <package>/output/                                      (written by the run)

Set AGRI_DATA to point somewhere else if you prefer.

The optional block at the bottom registers DLL directories bundled inside
numpy/scipy/sklearn.  It is a no-op on a normal install and only matters on
machines without the Visual C++ / OpenMP runtimes, where lightgbm would
otherwise fail to load.

Usage:  import env   # must be the first project import
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DATA = os.environ.get("AGRI_DATA") or os.path.join(PKG, "data")
OUTDIR = os.path.join(PKG, "output")


def _register_dll_dirs():
    if not hasattr(os, "add_dll_directory"):
        return
    for mod in ("numpy", "scipy", "sklearn"):
        try:
            base = os.path.dirname(__import__(mod).__file__)
        except Exception:
            continue
        for d in glob.glob(os.path.join(base, ".libs")) + \\
                 glob.glob(os.path.join(os.path.dirname(base), mod + ".libs")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def bootstrap():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    _register_dll_dirs()
    os.makedirs(OUTDIR, exist_ok=True)
    import common
    common.DATA = DATA
    return common


common = bootstrap()
'''

REQUIREMENTS = """numpy==2.5.3
pandas==3.0.1
scipy==1.18.1
scikit-learn==1.9.1
lightgbm==4.7.0
"""

README = """# 팜모니 — 정형데이터 미션 재현 패키지

제1회 전국 농과계 대학 농업 AI 경진대회, 정형데이터 미션.
문제설명서 8.2절이 요구하는 코드, 설정, 실행 안내, 자료 사용 내역을 담았습니다.

제출 답안 파일은 `submission_03.csv` 이며, 아래 절차로 바이트 단위까지 다시
만들어집니다.

---

## 1. 폴더 구성

```
├─ README.md                 이 문서 (실행 안내 + 자료 사용 내역)
├─ requirements.txt          라이브러리 버전
├─ submission_03.csv         제출한 답안 (1,440행)
├─ config/
│  └─ manifest_03.json       입력 파일 해시, 답안 해시, 피처 목록, 모델 파라미터
├─ data/                     원본 CSV 3개를 여기에 넣습니다 (배포본에는 비어 있음)
└─ code/
   ├─ env.py                 경로 설정
   ├─ common.py              원본 로딩, 컬럼 정의, 블록 CV 분할
   ├─ features_v2.py         인과적 피처 생성기 (물리 기반)
   ├─ feat_new.py            추가 파생 블록 (이슬점, 구동기 이벤트 등)
   ├─ feat_temp74.py         온도 피처 선택 (prev_day·mem_long 제거)
   ├─ feat_lib.py            그룹 정의, 온실-일 블록 부트스트랩
   ├─ harness.py             검증용 교차검증 하네스
   ├─ make_submission_v3.py  ★ 최종 답안 생성
   ├─ causality_test.py      규정 검사 (기존 피처)
   ├─ causality_new.py       규정 검사 (추가 파생 블록)
   └─ causality_shrink.py    규정 검사 (후처리 단계)
```

---

## 2. 실행 안내

### 2.1 실행 환경

- Python 3.12.14 (Windows 10, x64)
- 라이브러리 버전은 `requirements.txt` 와 `config/manifest_03.json` 에 기록

```
numpy 2.5.3 / pandas 3.0.1 / scipy 1.18.1 / scikit-learn 1.9.1 / lightgbm 4.7.0
```

GPU 를 사용하지 않습니다. 전 과정이 CPU 에서 동작합니다.

### 2.2 설치

```bash
python -m venv .venv
.venv\\Scripts\\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 원본 자료 배치

대회 원본 CSV 3개를 `data/` 에 넣습니다. 저작권상 배포본에는 포함하지
않았습니다.

```
data/train_X.csv
data/train_y.csv
data/test_X.csv
```

다른 위치에 두려면 환경변수 `AGRI_DATA` 에 해당 폴더 경로를 지정합니다.

### 2.4 실행

```bash
cd code
python make_submission_v3.py
```

`output/submission_03.csv` 와 `output/manifest_03.json` 이 만들어집니다.
동봉한 `submission_03.csv` 와 바이트 단위로 같아야 합니다.

```bash
# 대조
python -c "import hashlib;print(hashlib.sha256(open('../output/submission_03.csv','rb').read()).hexdigest())"
```

기대 해시는 `config/manifest_03.json` 의 `submission_sha256` 값입니다.

### 2.5 규정 검사 (선택)

```bash
cd code
python causality_test.py      # 기존 피처 125개
python causality_new.py       # 추가 파생 47개
python causality_shrink.py    # 후처리 단계
```

세 검사 모두 `ALL PASS` 가 나와야 합니다. 각각 미래 입력을 교란했을 때 이전
시각의 값이 불변인지, 정답 컬럼을 참조하지 않는지, 행 순서에 독립인지를
확인합니다.

---

## 3. 모델과 설정

모델 이진 파일은 포함하지 않았습니다. 학습 파이프라인이 결정론적이라
재학습하면 같은 값이 복원되기 때문입니다. 실제로 두 번 생성해 답안 파일이
바이트 단위로 동일함을 확인했습니다.

결정론을 보장하는 장치는 세 가지입니다. LightGBM 에 `deterministic=True`
와 `force_col_wise=True` 를 지정했고, 모든 시드를 7, 101, 2024 로 고정했으며,
ExtraTrees 는 예측을 단일 스레드로 수행합니다. 병렬 합산은 부동소수점 덧셈
순서를 바꾸어 소수 6째 자리를 흔듭니다.

### 3.1 배지 온도 (sub_temp)

93개 피처를 사용합니다. `features_v2` 의 온도 뷰 98개에서 전일 집계 18개와
장기 지수평활 6개를 제거하고, 이슬점 5개와 구동기 이벤트 경과시간 14개를
더한 구성입니다.

| 구성원 | 가중치 |
|---|---|
| LightGBM (huber, 1200트리, num_leaves 63) | 0.65 |
| Ridge (alpha=100, 표준화) | 0.25 |
| Nystroem(gamma=0.005, 500성분) + Ridge(alpha=1) | 0.10 |

각 구성원을 시드 3개로 학습해 평균합니다.

### 3.2 배지 EC (sub_ec)

14개 피처를 사용합니다. 평가자료에 존재하는 입력 14개 중 외부 기상 4개를
제외한 10개에, 작기 일차와 시각 표현 4개를 더한 구성입니다.

| 구성원 | 가중치 |
|---|---|
| ExtraTrees (600트리, min_samples_leaf 1) | 0.60 |
| LightGBM (tweedie, variance_power 1.5, 800트리) | 0.30 |
| MLP (128-64, alpha 1e-2, early stopping) | 0.10 |

블렌드 후 두 단계의 후처리를 적용합니다. 먼저 같은 온실-일 안에서 확장평균
쪽으로 0.5 만큼 수축합니다. 이는 모델이 만들어내는 하루 안의 변동이 실제
변동과 상관 0.2 수준에 불과하다는 진단에 따른 것입니다. 그다음 학습 정답의
관측 범위인 [0.062, 3.460] 으로 자릅니다.

수축은 h 시의 값을 계산할 때 같은 온실-일의 0시부터 h시까지의 예측만
사용하므로, 이후 구간의 입력에 의존하지 않습니다. `causality_shrink.py` 가
이를 기계적으로 확인합니다.

### 3.3 학습 자료 범위

두 목표 모두 평가 대상 온실인 F13 과 F47 의 라벨 9,600행만 사용합니다.
나머지 49개 온실의 온도 라벨은 1℃ 단위로 반올림되어 있어, 포함하면 검증
성능이 나빠지는 것을 확인하고 제외했습니다.

---

## 4. 자료 사용 내역

문제설명서 8.1절에 따라 사용한 자료와 도구를 밝힙니다.

| 구분 | 사용 여부 | 내용 |
|---|---|---|
| 외부 데이터 | 사용하지 않음 | 대회 제공 CSV 3개 외에 어떤 자료도 사용하지 않았습니다 |
| 공개 사전학습 모델 | 사용하지 않음 | 모든 모델을 대회 자료만으로 처음부터 학습했습니다 |
| 자체 생성 자료 | 사용하지 않음 | 합성 데이터를 만들지 않았습니다 |
| 생성형 AI | **사용함** | 아래 4.1 참조 |

평가 예측 과정에서 외부 API 나 원격 추론 서비스를 사용하지 않으며,
`test_X` 를 외부로 전송하지 않고, 외부 데이터를 조회하거나 내려받지
않습니다. 전 과정이 로컬에서 완결됩니다.

### 4.1 생성형 AI 사용 내역

- **도구**: Anthropic 의 Claude (Claude Code 환경)
- **이용조건**: Anthropic 상용 이용약관에 따른 유료 구독. 산출물의 사용
  권한은 이용자에게 있습니다.
- **활용 방법**: 코드 작성, 실험 설계, 교차검증 방법론 검토, 결과 해석에
  사용했습니다. 구체적으로는 피처 그룹 제거 실험과 모델 계열 탐색 스크립트
  작성, 온실-일 블록 부트스트랩 기반 유의성 검정 설계, 평가자료의 블록 배치를
  재현하는 교차검증 구성에 활용했습니다.
- **활용하지 않은 범위**: 대회 자료를 외부로 전송하지 않았습니다. 모델
  가중치나 예측값을 생성형 AI 로 만들지 않았으며, 모든 수치는 동봉한 코드를
  로컬에서 실행해 산출했습니다.

### 4.2 문헌 근거

`features_v2.py` 의 물리 기반 파생변수는 아래 문헌의 형태를 따랐습니다.
코드나 데이터를 가져오지 않았고, 수식 형태만 참고했습니다.

- Baille et al. (1994), 온실 증산 추정의 단순화 Penman-Monteith 형태.
  `transp_pm` 의 계수 범위(a 0.12~0.67, b 14~37e-3)를 참고했습니다.
- Magnus 식. `feat_new.py` 의 이슬점 계산에 사용한 표준형입니다.

---

## 5. 검증 요약

평가자료의 블록 배치를 그대로 옮긴 교차검증(`harness.py`)에서의 값입니다.
배치를 달리한 두 세트에서 모두 측정했습니다.

| 목표 | 기존 제출 구성 | 본 패키지 구성 |
|---|---|---|
| sub_temp | 0.8603 / 0.7862 | **0.8212 / 0.7671** |
| sub_ec | 0.3649 / 0.3372 | **0.2718 / 0.2376** |

교차검증 절대값은 실제 채점값보다 비관적입니다. 학습 입력에만 인위적
복원·변형이 포함되어 있고 평가 입력에는 없기 때문입니다(문제설명서 4절).
1회차 제출에서 교차검증 0.8603 / 0.3649 에 대해 실제 채점은 0.7450 / 0.2442
였습니다.
"""


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    pkg = os.path.join(STAGE, TEAM)
    for sub in ("code", "config", "data", "output"):
        os.makedirs(os.path.join(pkg, sub), exist_ok=True)

    for f in FROM_RESEARCH:
        shutil.copy2(os.path.join(HERE, f), os.path.join(pkg, "code", f))
    for f in FROM_CLAUDE:
        shutil.copy2(os.path.join(CODE_SRC, f), os.path.join(pkg, "code", f))

    # causality_test.py predates the packaging layout: it imports `common`
    # without importing `env` first, so common.DATA keeps its default and a
    # reviewer running it directly gets FileNotFoundError.  Inject the
    # bootstrap import rather than editing the original in Claude/code.
    ct = os.path.join(pkg, "code", "causality_test.py")
    with open(ct, encoding="utf-8") as fh:
        body = fh.read()
    needle = "import sys\n"
    assert needle in body, "causality_test.py layout changed"
    patched = body.replace(
        needle,
        "import sys\n\nimport env  # noqa: F401  packaged: resolves DATA "
        "and sys.path\n", 1)
    assert "import env" in patched
    with open(ct, "w", encoding="utf-8") as fh:
        fh.write(patched)
    with open(os.path.join(pkg, "code", "env.py"), "w", encoding="utf-8") as fh:
        fh.write(PORTABLE_ENV)
    with open(os.path.join(pkg, "requirements.txt"), "w", encoding="utf-8") as fh:
        fh.write(REQUIREMENTS)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README)
    with open(os.path.join(pkg, "data", "원본CSV를_여기에_넣으세요.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("train_X.csv, train_y.csv, test_X.csv 세 개를 이 폴더에 넣으세요.\n")

    src = os.path.join(HERE, "submissions")
    shutil.copy2(os.path.join(src, "submission_03.csv"),
                 os.path.join(pkg, "submission_03.csv"))
    shutil.copy2(os.path.join(src, "manifest_03.json"),
                 os.path.join(pkg, "config", "manifest_03.json"))

    zpath = os.path.join(src, TEAM + ".zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(pkg):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, STAGE))
                n += 1
    print("staged  %s" % pkg)
    print("wrote   %s" % zpath)
    print("        %d files, %.1f KB" % (n, os.path.getsize(zpath) / 1024.0))


if __name__ == "__main__":
    main()
