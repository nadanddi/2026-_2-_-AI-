# 팜모니 정형데이터 — TabPFN EC 혼합 검토용 재현 패키지

**Built with PriorLabs-TabPFN**

이 패키지는 온도 5회차 제출 구성과 EC 3회차 기준 모델 80% +
TabPFN-v2 회귀 모델 네 문맥 평균 20%를 결합한 **미제출 검토 후보**를
오프라인에서 재생성한다. 3회차 EC와 5회차 온도는 실제 플랫폼에
제출한 구성이고, TabPFN 혼합은 공개 정답 교차검증에서만 평가됐다.
이 패키지의 후보 CSV에 대한 비공개 RMSE는 알려져 있지 않다.

## 입력과 실행

`data/`에 대회 제공 `train_X.csv`, `train_y.csv`, `test_X.csv`를 넣는다.
원본 CSV는 패키지에 동봉하지 않았다. Python 3.12.14, Windows x64,
CPU 환경에서 `pip install -r requirements.txt` 후 다음을 실행한다.

```text
cd code
python reproduce_candidate.py
```

결과는 `output/candidate_temp06_ec_tabpfn_v2_cpu.csv`에 생긴다.
출력 파일이 있으면 덮어쓰지 않고 중단한다. 로컬에서 이미 설치한
의존성 묶음을 쓰는 경우에만 `AGRI_PACKAGE_DEPS_ROOT`를 해당 묶음
경로로 지정할 수 있다. 일반 설치에는 이 변수가 필요 없다.

모델 파일 `model/tabpfn-v2-regressor.ckpt`를 패키지에 포함했다.
SHA-256은
`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`이다.
코드는 이 파일의 해시와 캐시 경로를 확인한 뒤 실행한다. 원격 API,
원격 추론, 다운로드, 평가 입력의 외부 전송은 사용하지 않는다.

## 구성과 검증

온도는 기존 5회차 구성의 가중치 적용 물리·잔차 LightGBM / Ridge /
Nystroem 혼합이다. EC 기준은 ExtraTrees 60%, LightGBM Tweedie 30%,
MLP 10%를 시드 7·101·2024에서 평균한다. TabPFN은 38개 인과적
특징으로 훈련행 2,000개씩을 시드 1~4로 추출하고 ModelVersion.V2,
CPU float32, n_estimators=4로 각각 예측한 뒤 단순 평균한다.
최종 EC는 기준 원시 예측 80% + TabPFN 20%에 당일 누적 수축 0.5와
훈련 EC 범위 제한을 적용한다.

생성기는 출력 전에 `reference/submission_06.csv`의 온도,
`reference/submission_04.csv`의 EC 기준 예측을 전 행 소수 6자리로
대조한다. 생성 후 `reference/candidate_temp06_ec_tabpfn_v2_cpu.csv`와
전 행 6자리 값을 대조한다. 세 검사가 모두 통과하면
`output/manifest_tabpfn_v2_cpu.json`에 입력·가중치·출력 해시와
라이브러리 버전을 기록한다. 패키지의 `reference/` 파일은 검사에만
사용하고 예측값의 입력으로 사용하지 않는다.

개발 중 실제 test_X 1,440행에 대해 F13·F47 각 네 절단점의 미래
입력 교란과 다른 온실 교란에서 앞선 특징 불변성을 확인했다.
네 문맥 각각에서 행 순서 역전과 F13 중간 절단점의 미래 입력
교란에도 앞선 예측이 동일했고, 문맥1 독립 재적합도 동일했다.
이 검사의 설정과 결과는 개발 기록에 보존했고, 패키지에는
이전 온도 구성의 규정 검사 코드도 동봉한다.

## 자료와 도구 출처

대회 원본 CSV 외의 **추가 학습 표**는 사용하지 않았다. 공개
사전학습 모델 [Prior Labs TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg)를
EC 예측에 사용했다. 모델 가중치는 별도의 [Prior Labs License v1.1](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt)를
따르며, 전문은 `model/LICENSE.txt`에 동봉했다. TabPFN 코드 9.0.0의
라이선스는 Apache-2.0이고 설치 과정에서 제공된다. 가중치 사용에
대한 출처 표기는 이 문서 상단과 모델 파일 설명에 포함했다.

분석·코드 작성에는 Anthropic Claude와 OpenAI Codex를 사용했다.
생성형 AI는 모델 설계·코드·검증 방법을 돕는 데 사용했으며,
후보 예측 숫자는 오프라인 로컬 코드에서 계산한다. 이전 온도 구성의 문헌과
추가 분석 내역은 `README_previous_07.md`에 보존했다.

이 패키지는 대회 문제설명서 8.2절의 재현 검토를 위한 로컬 초안이다.
플랫폼에 제출하거나 최종 성적을 주장하지 않는다.
