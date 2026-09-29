# -*- coding: utf-8 -*-
"""Build the reproduction ZIP for submission_06 (platform round 5).

Shared pieces (portable env.py, requirements, the causality_test patch) are
imported from build_package_v4 unchanged.  Stages into local/package_v6 and
writes submissions/팜모니_정형데이터_재현패키지_06.zip; refuses to overwrite.

Audit fixes carried in (research/감사결과_2026-09-26.md): OpenAI Codex added
to the generative-AI disclosure, CRLF / hash note, overwrite guard in the
generator, train_flags_v6.py computed straight from the raw CSVs, stronger
rule check causality_v6.py.

Run:  cd research && PYTHONPATH="" <python> build_package_v6.py
"""
import os
import shutil
import zipfile

import build_package_v4 as B4

HERE = B4.HERE
STAGE = os.path.join(HERE, "local", "package_v6")
TEAM = B4.TEAM
FILES = B4.FROM_RESEARCH_V4 + ["train_flags_v6.py", "make_submission_v6.py", "causality_v6.py"]

README = """# 팜모니 — 정형데이터 미션 재현 패키지 (submission_06, 제출 5회차)

제1회 전국 농과계 대학 농업 AI 경진대회, 정형데이터 미션.
문제설명서 8.2절이 요구하는 코드, 설정, 실행 안내, 자료 사용 내역을 담았습니다.
제출 답안 `submission_06.csv` 는 아래 절차로 다시 만들어집니다.

---

## 1. 폴더 구성

```
├─ README.md
├─ requirements.txt
├─ submission_06.csv          제출한 답안 (1,440행)
├─ config/manifest_06.json    입력 해시, 답안 해시, 피처 목록, 모델 파라미터, 학습 가중치 설정
├─ reference/submission_04.csv  3회차 제출본. 변화량 출력용
├─ data/                      원본 CSV 3개를 넣는 곳 (배포본은 비어 있음)
└─ code/
   ├─ env.py, common.py, harness.py, feat_lib.py
   ├─ features_v2.py          인과적 피처 생성기 (물리 기반)
   ├─ feat_new.py             이슬점, 구동기 이벤트 등
   ├─ feat_temp74.py          온도 피처 선택
   ├─ features_v4.py          자정 재시작 필터, 물리 기준선 입력
   ├─ fp_features.py          출처 지문
   ├─ train_flags_v6.py       학습 행 가중치 (학습 입력만 사용)
   ├─ make_submission_v3.py   모델 정의·후처리 함수
   ├─ make_submission_v4.py   EC 블렌드 가중치
   ├─ make_submission_v6.py   ★ 최종 답안 생성
   └─ causality_test.py, causality_new.py, causality_v4.py,
      causality_shrink.py, causality_v6.py   규정 검사
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
python make_submission_v6.py
```
`output/submission_06.csv` 가 만들어집니다. 출력 폴더에 같은 이름의 파일이 이미 있으면
덮어쓰지 않고 중단합니다(다시 돌릴 때는 output/ 을 비우세요).

**해시 비교 주의**: 답안 CSV는 Windows에서 pandas 기본값으로 저장되어 줄바꿈이 CRLF입니다.
`config/manifest_06.json` 의 `submission_sha256` 은 Windows 기준입니다. Linux·macOS에서
재현하면 줄바꿈 차이로 해시가 다를 수 있으니, 이 경우 두 파일을 읽어 수치를 비교하세요
(소수 6자리까지 같아야 함).

### 2.5 규정 검사
```bash
cd code
python causality_test.py
python causality_new.py
python causality_v4.py
python causality_shrink.py
python causality_v6.py
```
모두 `ALL PASS` 여야 합니다. causality_v6.py 는 모델에 들어가는 150열 전부에 대해
(1) 온실별 여섯 절단 시점 이후 입력을 10배 확대·잡음·33% 결측으로 바꿔도 이전 행이
비트 단위로 같은지, (2) 다른 온실 입력을 바꿔도 해당 온실 행이 같은지, (3) train_y 를
섞어도 피처가 같은지, (4) 학습 행 가중치가 test_X 와 무관하고 복원 행 판정이 과거
방향인지를 확인합니다.

## 3. 모델과 설정

모델 이진 파일은 포함하지 않았습니다. 결정론적 파이프라인이라 재학습하면 같은 값이
나옵니다(LightGBM deterministic, 시드 7·101·2024 고정, ExtraTrees 예측 단일 스레드).

### 3.1 배지 온도

피처 135개와 모델 구성은 3회차 제출(submission_04)과 같습니다.

```
예측 = 0.65 × (선형 물리 기준선 + LightGBM(huber)이 학습한 잔차)
     + 0.25 × Ridge(alpha=100) + 0.10 × Nystroem + Ridge
```

바뀐 것은 **학습 행 가중치**뿐입니다(train_flags_v6.py). 학습 입력에는 복원된 행과 잡음이
주입된 날이 섞여 있고, 평가 입력에는 이런 흔적이 거의 없습니다. 두 가지를 가중치 0.2로
낮췄습니다(9,600행 중 2,341행).

- 복원 행 60개: 밤(19~06시)에 실내온도가 실외보다 3℃ 이상 낮음 / 실내온도가 6시간 이상
  그대로(지난 시각 방향으로만 셈) / 자정이 아닌 밤에 CO₂ 공급 없이 CO₂가 150 이상 급변
- 잡음 낀 날 96일: 하루 안에서 CO₂ 시간 차분의 1시차 자기상관(잡음 낀 날 중앙값 −0.12,
  나머지 학습일 0.60, 평가일 0.46)과 CO₂ 2차 차분 크기로 학습일끼리만 순위를 매겨 상위
  25%. 추운 행(3시간 평활 8℃ 미만)이 있는 날은 제외(희소한 한랭 학습 자료 보존)

가중치는 학습 입력만으로 계산하며 test_X, 라벨을 쓰지 않습니다.

### 3.2 배지 EC
3회차 구성에서 ExtraTrees 입력의 `day` 열 하나만 뺐습니다. ExtraTrees(13열 + 출처 지문)
0.60, LightGBM tweedie(14열) 0.30, MLP(14열) 0.10, 일내 수축 0.5, 범위 [0.062, 3.460] 제한.

### 3.3 인과성
모든 피처는 같은 온실 기록의 해당 시각과 이전 입력만 사용합니다(2.5 검사).

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

- **도구 1**: Anthropic Claude (Claude Code)
  - 이용조건: Anthropic 이용약관에 따른 유료 구독. 산출물 사용 권한은 이용자에게 있음.
  - 활용 방법: 코드 작성, 탐색적 자료 분석, 실험 설계, 검증 방법론 검토, 결과 해석.
- **도구 2**: OpenAI Codex
  - 이용조건: OpenAI 이용약관에 따른 유료 구독. 산출물 사용 권한은 이용자에게 있음.
  - 활용 방법: 별도 분석 계열(탐색적 분석, 교차검증, 모델 실험 코드 작성). 그 결과 중
    커튼 피처 부호 정정, ExtraTrees 잎 크기 설정, EC 모델 구성(ExtraTrees 중심 블렌드)을
    본 패키지 구성에 반영.
- **활용하지 않은 범위**(두 도구 공통): 대회 자료 외부 전송 없음. 예측값을 생성형 AI 로
  만들지 않았고, 모든 수치는 동봉한 코드를 로컬에서 실행해 산출.

문헌: Baille et al. (1994) 증산식 형태(features_v2.py), Magnus 식 이슬점(feat_new.py).

## 5. 검증 요약 (온도, 3회차 구성 대비)

겹치지 않는 폴드(학습일마다 정확히 한 번 제외)와, 추운 날을 통째로 빼는 외삽 검증 세 가지
임계로 전체 행을 채점했습니다. 괄호는 온실-날 블록 짝지은 부트스트랩 95% 구간입니다.

| 검증 | 3회차 구성 | 본 구성 | 차이 |
|---|---|---|---|
| 진단 폴드(전체 학습일) | 0.7061 | 0.6965 | −0.0096 [−0.0206, +0.0003] |
| 외삽, 8℃ 미만 날 제외 | 1.0028 | 1.0004 | −0.0024 (유의하지 않음) |
| 외삽, 10℃ 미만 날 제외 | 0.9090 | 0.8806 | −0.0284 [−0.0525, −0.0049] |
| 외삽, 12℃ 미만 날 제외 | 0.9270 | 0.8858 | −0.0412 [−0.0559, −0.0276] |

EC의 `day` 제거는 검증끼리 결론이 엇갈려(구간 검증 개선, 블록 교차검증 악화) 실험으로
제출했습니다.
"""


def main():
    src = os.path.join(HERE, "submissions")
    zpath = os.path.join(src, TEAM + "_06.zip")
    assert not os.path.exists(zpath), "refusing to overwrite an existing package: %s" % zpath
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

    shutil.copy2(os.path.join(src, "submission_06.csv"), os.path.join(pkg, "submission_06.csv"))
    shutil.copy2(os.path.join(src, "manifest_06.json"), os.path.join(pkg, "config", "manifest_06.json"))
    shutil.copy2(os.path.join(src, "submission_04.csv"),
                 os.path.join(pkg, "reference", "submission_04.csv"))

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
