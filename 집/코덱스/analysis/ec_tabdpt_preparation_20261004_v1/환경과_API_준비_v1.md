# TabDPT v1.3.0 환경·API 실행 준비 v1

작성: 2026-10-04, 집/코덱스. **읽기 조사와 비활성 초안 준비만 완료했다.** family 등록, 설치, 신규 checkpoint 다운로드, 실제 fit/predict, 데이터 업로드는 0이다. EC 점수 개선 여부는 미검증이다.

## 공식 버전·API 고정

공식 [v1.3.0 release](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.0)는 2026-09-08이며 [commit 97e5494](https://github.com/layer6ai-labs/TabDPT-inference/commit/97e5494431e9527c7edb31cb4dcfc5f00b232fdf)에 고정한다. 원문 읽은 범위는 tagged estimator.py, regressor.py, utils.py, model.py, pyproject.toml과 공개 release/commit 페이지다. 공식 test_inference.py는 읽기에 실패했으므로 그 테스트를 확인했다고 주장하지 않는다.

[regressor.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/regressor.py)의 API는 `from tabdpt.regressor import TabDPTRegressor`이다. constructor와 predict 인자를 분리한 [adapter_draft_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/adapter_draft_v1.py)를 저장했다. predict는 `n_ensembles=8, context_size=512, batch_size=8, seed=..., output_type="mean"`을 받는다. mean은 각 앙상블의 점예측 평균이다.

[estimator.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/estimator.py)의 constructor는 weight 경로를 생략하면 HF 다운로드를 호출한다. 초안은 검증된 로컬 `model_weight_path`를 필수로 넘기고 offline 설정을 둔다. fit은 train으로 mean imputer/StandardScaler를 맞추고 context를 저장한다. 이는 checkpoint 파라미터를 gradient로 재학습하는 fit은 아니지만, 아직 이 작업에서도 실제 fit하지 않았다.

## 현재 환경과 최소 분리 경로

아래는 패키지 import 결과가 아니라 .dist-info 메타데이터를 stdlib로 읽은 결과다. DLL 로드 가능성은 별도 확인해야 한다. [inventory_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/inventory_v1.json), [읽기 스크립트](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/inventory_readonly_v1.py)에 근거와 경로를 보관했다.

| 패키지 | 현재 metadata | tagged 요구 | 준비 판단 |
|---|---:|---|---|
| Python | 3.12.10 / AMD64 Windows | ≥3.10 | 인터프리터 범위 충족 |
| numpy | 2.5.3 | ≥1.25,<3 | 기존 core 재사용 제안 |
| scikit-learn | 1.9.1 | ≥1.4,<2 | 기존 core 재사용 제안 |
| scipy | 1.18.1 | ≥1.9,<2 | 기존 core 재사용 제안 |
| torch | **2.14.0** | ≥2.6,<3 | 기존 extra 재사용 제안 |
| safetensors | **0.8.0** | ≥0.5.3,<1 | 기존 extra 재사용 제안 |
| tqdm | 4.70.1 | ≥4.38,<5 | 기존 extra 재사용 제안 |
| huggingface-hub | **2.0.0** | ≥0.33.2,<2.0 | 기존 버전은 범위 밖; 별도 overlay 필요 |
| faiss-cpu / omegaconf / tabdpt | metadata 없음 | 각각 ≥1.11,<1.13 / ≥2.1.1,<3 / 고정1.3.0 | 별도 overlay 필요 |

요구 범위는 공식 [pyproject.toml](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/pyproject.toml) 기준이다. 범위 충족을 실제 import 성공 또는 모델 호환성으로 해석하지 않는다.

**제안 경로**는 `C:\work\farmai\집\코덱스\local\tabdpt130_cpu_v1\site`다. 이 경로에 TabDPT의 고정 VCS commit, faiss-cpu=1.12.0, OmegaConf와 Hub의 호환 버전 및 필요한 전이 의존성만 놓는다. 큰 torch/numpy/scipy/sklearn/safetensors를 다시 설치하지 않는 최소 overlay 제안이다. 기존 R3/PFN process와 공용 .analysis-tools 패키지는 변경하지 않는다. 깨끗한 별도 Python worker에서만 overlay를 앞에 넣는다.

호환 runtime 후보 pin으로 Hub=0.36.0, OmegaConf=2.3.0을 제안할 수 있다. 공식 [Hub 0.36.0 metadata](https://pypi.org/pypi/huggingface-hub/0.36.0/json)는 Python≥3.8과 filelock/fsspec/packaging/PyYAML/requests/tqdm/typing-extensions 및 플랫폼 조건의 hf-xet를 요구한다. [OmegaConf 2.3.0 metadata](https://pypi.org/pypi/omegaconf/2.3.0/json)는 antlr4-python3-runtime==4.9.*와 PyYAML≥5.1을 요구한다. **전이 의존성 현재 버전·정확한 설치 lock는 아직 확정하지 않았다.** 따라서 이 목록을 완성된 pip 설치 명령으로 간주하지 않는다. 설치 담당자가 새 lock 파일과 배포물 hash를 확정한다.

현재 프로젝트의 [env.py](C:/work/farmai/집/클로드/research/env.py)는 core .libs를, [env_extra.py](C:/work/farmai/집/클로드/research/env_extra.py)는 `.analysis-tools/msvc`와 `.analysis-tools/extra/torch/lib`를 DLL 검색 경로로 등록하도록 작성되어 있다. 따라서 **env → env_extra 순서의 DLL 초기화가 필요하다.** 새 FAISS overlay의 .libs도 포함해야 하며 실제 wheel 내부 DLL 위치는 설치 후 확인한다. [Python 공식 문서](https://docs.python.org/3.12/library/os.html#os.add_dll_directory)에 따르면 반환 핸들을 닫으면 경로가 제거된다. 새 [runtime_probe_draft_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/runtime_probe_draft_v1.py)는 worker 동안 DLL 핸들을 보관하도록 준비했다. 공용 bootstrap은 수정하지 않았다.

아래는 **담당자의 공개 접근·license·설치 검토 후 실행할 환경 import 검사 명령**이며 이 작업에서는 실행하지 않았다. checkpoint constructor/fit/predict/CV를 호출하는 명령이 아니다.

```powershell
$env:PYTHONPATH = ""
python -X utf8 -B "C:\work\farmai\집\코덱스\analysis\ec_tabdpt_preparation_20261004_v1\runtime_probe_draft_v1.py" --runtime-root "C:\work\farmai\집\코덱스\local\tabdpt130_cpu_v1\site" --check-imports
```

probe는 package version/실제 module 경로, DLL 경로, 공식 API 서명을 출력하고 TabDPT `direct_url.json`의 **전체 VCS commit**이 다르면 멈춘다. PyPI 버전 번호만 같고 VCS commit을 입증하지 못하는 설치는 이 검사에서 통과하지 않는다. probe default 실행은 명세만 출력한다. 실제 CV runner는 담당자가 동일 baseline fold에 연결해 새 버전으로 작성해야 한다.

## Windows FAISS 공식 PyPI 파일 확인

공식 [faiss-cpu 1.12.0 PyPI files](https://pypi.org/project/faiss-cpu/1.12.0/#files)에 **faiss_cpu-1.12.0-cp312-cp312-win_amd64.whl**이 있다. CPython3.12 / Windows x86-64, 18.2MB, 2025-08-13 업로드이며 SHA256은 **6b8012353d50d9bc81bcfe35b226d0e5bfad345fdebe0da31848395ebc83816d**이다. 프로젝트 요구 ≥1.11,<1.13 범위다.

PyPI license expression은 MIT AND BSD-3-Clause다. 공식 PyPI 배포 파일의 존재를 확인한 것이며 Meta가 직접 빌드한 파일이라고 주장하지 않는다. wheel 다운로드·설치·DLL import는 하지 않았다. 실제 설치 환경에서의 성공은 미확인이다.

## 공개 weight metadata와 미확인 사항

공식 [파일 blob](https://huggingface.co/Layer6/TabDPT/blob/main/tabdpt1_3.safetensors)과 [weight 추가 commit](https://huggingface.co/Layer6/TabDPT/commit/a5ca6e01c0fa09ec68c73e958e5199d1932abb3a)의 LFS pointer에서 파일명·size·SHA256을 대조했다.

| 항목 | 읽기 확인 |
|---|---|
| repo | Layer6/TabDPT |
| 고정 revision | a5ca6e01c0fa09ec68c73e958e5199d1932abb3a |
| 파일 | tabdpt1_3.safetensors |
| bytes | 252233296 |
| SHA256 | 97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb |
| Xet hash | 6ca36e156a9b643fa1211de0ad6589cb11030cd1d98fed8ed2eb4fc773041a29 |
| 다운로드 / 로컬 hash 검사 | 0 / 미실행 |

공식 [main 모델 카드](https://huggingface.co/Layer6/TabDPT/raw/main/README.md)의 YAML은 apache-2.0이고, [main LICENSE](https://huggingface.co/Layer6/TabDPT/blob/main/LICENSE)는 Apache License 2.0이다. 카드 본문은 v1.1 설명을 남겨 현재 checkpoint의 구체 환경 설명에는 사용하지 않았다.

익명 공식 페이지에서 파일 metadata가 보이고 동의 버튼을 누르지 않았다. **HF API의 gated/private flag 조회와 해당 revision의 LICENSE/card 읽기는 실패했다. gated=false·private=false는 확인하지 못했다.** main license 표기를 상업 조건 자동승인이나 별도 계약 동의 완료로 해석하지 않는다. 가중치 취득 전 담당자가 공개 접근과 적용 license를 확인한다. 현재 상태는 [source_metadata_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/source_metadata_v1.json)에 null/미확인으로 기록했다.

## Query별 독립성과 단독·배치 기대

이 부분은 **공식 소스 구조에 따른 판단**이다. 수치 감사 결과는 아니다.

[utils.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/utils.py)의 retrieval index는 standardized train float32의 IndexFlatL2다. 각 query는 그 index에서 512개 train 행을 검색한다. [regressor.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/regressor.py)는 각 query의 512행과 자기 1행만 ordinary batch의 한 원소로 배치한다. feature 순열 seed와 y 정규화 통계도 고정 train에서 정해진다.

[model.py](https://raw.githubusercontent.com/layer6ai-labs/TabDPT-inference/v1.3.0/src/tabdpt/model.py)의 입력은 (batch,context+1,feature)이며 transpose 이후 normalize/clip은 **context 축**과 eval_pos 이전 자료를 사용한다. attention K/V 역시 해당 batch 원소의 context까지만 사용한다. LayerNorm/RMSNorm은 마지막 embedding 축을 사용한다. 따라서 다른 query의 값·순서를 바꿔도 대상 query의 수학적 식은 바뀌지 않으며, 같은 train/seed의 raw mean은 단독과 batch에서 같아야 한다는 기대가 성립한다.

그러나 FAISS 동률 순서, 라이브러리 수치 연산, CPU kernel 차이까지 소스만으로 완전 동일성을 보증할 수 없다. `self.X_test`가 predict에서 갱신되므로 같은 object의 동시 thread 호출도 금지한다. 첫 raw 8개 query에 반복·단독·역순·타 query 변화 감사를 준비했고 raw atol=1e-6은 **사전 확정 제안**이다. fresh-fit 두 번의 일치, 최종 prefix와 전체 특징 인과성 검사는 담당자가 추가한다. 문맥이 train 전체로 전환되지 않도록 n_train>512를 확인한다.

최종 shrink는 같은 farm-day 현재·과거 예측을 사용한다. 그러므로 raw 단독 행과 달리 **최종 행 검사는 같은 과거 prefix를 유지해야** 한다. 다른 query 독립성을 이 후처리에 잘못 적용하면 의도된 과거 의존성을 오류로 오판할 수 있다.

## 준비 산출물과 실제 검증 범위

[preregistration_draft_v1.md](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/preregistration_draft_v1.md)에 단일 후보식, 고정 설정, family 미등록, 채택 규칙과 중단 조건을 적었다. baseline 소스 hashes와 FULL38 대조는 [static_audit_result_v1.json](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/static_audit_result_v1.json), 재현 코드는 [static_audit_v1.py](C:/work/farmai/집/코덱스/analysis/ec_tabdpt_preparation_20261004_v1/static_audit_v1.py)에 저장했다.

정적 검사에서는 adapter·inventory·static audit의 문법, 실행 비활성 관문 7개, heavy package import 0, 기존 core FULL38 순서 일치를 확인했다. probe 문법과 default 무 import 실행도 별도로 확인한다. 이 PASS는 실제 모델의 DLL/fit/predict/반복성 PASS와 다르다.

카탈로그 6.48에는 TabICL 2.2.0의 기존 CPU 선별 탈락이 있다. TabDPT 후보와 다른 기록이며 그 결과를 이번 모델 개선 근거로 쓰지 않는다. 카탈로그에서 TabDPT 등록은 읽기 검색에 잡히지 않았지만, 이미 이전 문헌 검토에서 조사한 모델이므로 이번에 처음 발견한 모델이라고 표현하지 않는다.

