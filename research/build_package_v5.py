# -*- coding: utf-8 -*-
"""Build the reproduction ZIP for submission_05 (platform round 4).

Shared pieces (portable env.py, requirements, the causality_test patch) are
imported from build_package_v4 unchanged.  Stages into local/package_v5 so the
round-3 staging folder is left alone, and writes
submissions/팜모니_정형데이터_재현패키지_05.zip.

Run:  cd research && PYTHONPATH="" <python> build_package_v5.py
"""
import os
import shutil
import zipfile

import build_package_v4 as B4

HERE = B4.HERE
STAGE = os.path.join(HERE, "local", "package_v5")
TEAM = B4.TEAM
FILES = B4.FROM_RESEARCH_V4 + ["features_v5.py", "make_submission_v5.py", "causality_v5.py"]

README = """# 팜모니 — 정형데이터 미션 재현 패키지 (submission_05, 제출 4회차)

제1회 전국 농과계 대학 농업 AI 경진대회, 정형데이터 미션.
문제설명서 8.2절이 요구하는 코드, 설정, 실행 안내, 자료 사용 내역을 담았습니다.
제출 답안 `submission_05.csv` 는 아래 절차로 바이트 단위까지 다시 만들어집니다.

---

## 1. 폴더 구성

```
├─ README.md
├─ requirements.txt
├─ submission_05.csv          제출한 답안 (1,440행)
├─ config/manifest_05.json    입력 해시, 답안 해시, 피처 목록, 모델 파라미터
├─ reference/submission_04.csv  직전 제출본. 자가 검사용(아래 2.4)
├─ data/                      원본 CSV 3개를 넣는 곳 (배포본은 비어 있음)
└─ code/
   ├─ env.py, common.py, harness.py, feat_lib.py
   ├─ features_v2.py          인과적 피처 생성기 (물리 기반)
   ├─ feat_new.py             이슬점, 구동기 이벤트 등
   ├─ feat_temp74.py          온도 피처 선택
   ├─ features_v4.py          자정 재시작 필터, 물리 기준선 입력
   ├─ fp_features.py          출처 지문
   ├─ features_v5.py          한랭 힌지
   ├─ make_submission_v3.py   모델 정의·후처리 함수 (v4, v5가 사용)
   ├─ make_submission_v4.py   직전 구성 (v5가 사용)
   ├─ make_submission_v5.py   ★ 최종 답안 생성
   └─ causality_test.py, causality_new.py, causality_v4.py,
      causality_v5.py, causality_shrink.py   규정 검사
```

## 2. 실행 안내

### 2.1 환경
Python 3.12.14 (Windows 10, x64), GPU 미사용.
numpy 2.5.3 / pandas 3.0.1 / scipy 1.18.1 / scikit-learn 1.9.1 / lightgbm 4.7.0

### 2.2 설치
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2.3 원본 자료
`data/` 에 train_X.csv, train_y.csv, test_X.csv 를 넣습니다(또는 환경변수 AGRI_DATA).

### 2.4 실행
```bash
cd code
python make_submission_v5.py
```
`output/submission_05.csv` 가 만들어지며 동봉본과 바이트 단위로 같아야 합니다(기대 해시는
`config/manifest_05.json` 의 `submission_sha256`). 실행 중 자가 검사가 두 가지를 확인합니다.
EC 열 전체와 온도 블렌드의 직전 구성 절반이 `reference/submission_04.csv` 와 소수 6자리까지
같아야 합니다.

### 2.5 규정 검사
```bash
cd code
python causality_test.py
python causality_new.py
python causality_v4.py
python causality_v5.py
python causality_shrink.py
```
다섯 검사 모두 `ALL PASS` 여야 합니다.

## 3. 모델과 설정

모델 이진 파일은 포함하지 않았습니다. 결정론적 파이프라인이라 재학습하면 같은 값이 나오며,
두 번 생성해 바이트 동일함을 확인했습니다(LightGBM deterministic, 시드 7·101·2024 고정,
ExtraTrees 예측 단일 스레드).

### 3.1 배지 온도

피처 135개는 직전 제출과 같습니다. 모델만 바뀌었습니다.

```
예측 = 0.5 × [직전 구성 블렌드] + 0.5 × [한랭 힌지 블렌드]
블렌드 = 0.65 × (선형 물리 기준선 + LightGBM(huber)이 학습한 잔차)
       + 0.25 × Ridge(alpha=100) + 0.10 × Nystroem + Ridge
```

두 블렌드의 차이는 선형 물리 기준선의 입력뿐입니다. 한랭 힌지 블렌드는 기준선에
`max(0, k − x)` (k = 8, 10, 12℃, x = 실내온도 3시간 지수평활과 원값) 6개를 더해,
추운 구간에서 기준선이 꺾일 수 있게 했습니다.

근거: 평가 기간은 학습 라벨보다 춥고(평가 행의 12~18%가 학습 범위 밖), 학습 라벨에서는
추울수록 배지가 공기보다 더 차갑습니다(3시간 평활 공기 기준 10℃ 이상 −0.6~−0.8℃,
6~8℃ −1.17℃). 직전 구성은 가장 추운 평가 행을 공기보다 0.41~0.54℃ 낮게만 예측합니다.

### 3.2 배지 EC
직전 제출(submission_04)과 동일합니다. ExtraTrees(14열 + 출처 지문) 0.60, LightGBM tweedie
0.30, MLP 0.10, 일내 수축 0.5, 범위 [0.062, 3.460] 제한.

### 3.3 인과성
모든 피처는 같은 기록의 해당 시각과 이전 입력만 사용합니다. 한랭 힌지는 인과적 지수평활의
점별 함수입니다.

### 3.4 학습 자료
두 목표 모두 F13·F47 라벨 9,600행.

## 4. 자료 사용 내역

| 구분 | 사용 여부 | 내용 |
|---|---|---|
| 외부 데이터 | 사용하지 않음 | 대회 제공 CSV 3개만 사용 |
| 공개 사전학습 모델 | 사용하지 않음 | 모든 모델을 처음부터 학습 |
| 자체 생성 자료 | 사용하지 않음 | 합성 데이터 없음 |
| 생성형 AI | **사용함** | 아래 |

평가 예측 과정에서 외부 API·원격 추론을 쓰지 않고 test_X 를 외부로 보내지 않습니다.

- **도구**: Anthropic Claude (Claude Code)
- **이용조건**: Anthropic 이용약관에 따른 유료 구독. 산출물 사용 권한은 이용자에게 있음.
- **활용 방법**: 코드 작성, 탐색적 자료 분석, 실험 설계, 검증 방법론 검토, 결과 해석.
- **활용하지 않은 범위**: 대회 자료 외부 전송 없음. 예측값을 생성형 AI 로 만들지 않았고,
  모든 수치는 동봉한 코드를 로컬에서 실행해 산출.

문헌: Baille et al. (1994) 증산식 형태(features_v2.py), Magnus 식 이슬점(feat_new.py).

## 5. 검증 요약 (직전 구성 대비, 시드 7)

평가 기간처럼 학습 범위보다 추운 날을 통째로 빼고 예측하게 한 외삽 검증을 주로 썼습니다.
이 검증은 직전 두 제출의 실제 점수 비율(0.844)을 0.858로 재현했습니다.

| 검증 | 직전 구성 | 본 구성 |
|---|---|---|
| 외삽 검증, 8℃ 미만 날 제외(264행) | 1.0028 | 0.9510 |
| 그중 8℃ 미만 행 | 0.899 (편향 +0.54) | 0.785 (편향 +0.22) |
| 외삽 검증, 7℃ 미만 날 제외(72행) | 0.7259 | 0.7551 |
| 블록 교차검증 A / B | 0.7850 / 0.7434 | 0.7825 / 0.7410 |
"""


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    pkg = os.path.join(STAGE, TEAM)
    for sub in ("code", "config", "data", "output", "reference"):
        os.makedirs(os.path.join(pkg, sub), exist_ok=True)
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), os.path.join(pkg, "code", f))
    for f in B4.FROM_CLAUDE:
        shutil.copy2(os.path.join(B4.CODE_SRC, f), os.path.join(pkg, "code", f))

    ct = os.path.join(pkg, "code", "causality_test.py")
    body = open(ct, encoding="utf-8").read()
    assert "import sys\n" in body
    body = body.replace("import sys\n", "import sys\n\nimport env  # noqa: F401  packaged: "
                        "resolves DATA and sys.path\n", 1)
    open(ct, "w", encoding="utf-8").write(body)

    with open(os.path.join(pkg, "code", "env.py"), "w", encoding="utf-8") as fh:
        fh.write(B4.PORTABLE_ENV)
    with open(os.path.join(pkg, "requirements.txt"), "w", encoding="utf-8") as fh:
        fh.write(B4.REQUIREMENTS)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README)
    with open(os.path.join(pkg, "data", "원본CSV를_여기에_넣으세요.txt"), "w", encoding="utf-8") as fh:
        fh.write("train_X.csv, train_y.csv, test_X.csv 세 개를 이 폴더에 넣으세요.\n")

    src = os.path.join(HERE, "submissions")
    shutil.copy2(os.path.join(src, "submission_05.csv"), os.path.join(pkg, "submission_05.csv"))
    shutil.copy2(os.path.join(src, "manifest_05.json"), os.path.join(pkg, "config", "manifest_05.json"))
    shutil.copy2(os.path.join(src, "submission_04.csv"),
                 os.path.join(pkg, "reference", "submission_04.csv"))

    zpath = os.path.join(src, TEAM + "_05.zip")
    assert not os.path.exists(zpath), "refusing to overwrite an existing package: %s" % zpath
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fnames in os.walk(pkg):
            for f in sorted(fnames):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, STAGE))
                n += 1
    print("staged  %s" % pkg)
    print("wrote   %s" % zpath)
    print("        %d files, %.1f KB" % (n, os.path.getsize(zpath) / 1024.0))


if __name__ == "__main__":
    main()
