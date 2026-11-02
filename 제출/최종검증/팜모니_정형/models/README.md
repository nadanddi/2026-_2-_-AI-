# models — 가중치와 저장 중간값

## 사전학습 가중치
| 파일 | SHA256 | 출처·용도 |
|---|---|---|
| `tabpfn-v2-regressor.ckpt` (44,390,977 B) | `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` | Prior Labs TabPFN v2 회귀(Hugging Face `Prior-Labs/TabPFN-v2-reg`, `tabpfn` 9.0.0 기본 가중치). 온도 NX의 TabPFN, EC 구성원 TabPFN, EC HR5에 씁니다. 같은 파일이 `code/pipeline_35e/EC/`에도 있습니다(EC 스크립트가 그 위치를 읽음). 온도 NX 재학습은 `TABPFN_MODEL_CACHE_DIR=<이 폴더>`로 이 파일을 쓰게 합니다(`code/retrain/full_retrain.py`가 설정). |

## 저장 중간값 (참조 테이블)
최종 과정(`code/pipeline_35e/checks_v35e.py produce`)이 읽는 저장 파일입니다. 모델 가중치 파일을 따로 저장하지 않고, 각 단계의 예측을 저장해 두었습니다. 모두 `code/retrain/full_retrain.py`가 원본 자료로 다시 만들 수 있습니다. 파일은 코드가 읽는 위치에 그대로 두었습니다.

| 위치(`code/pipeline_35e/` 기준) | SHA256 | 내용 | 다시 만드는 코드 |
|---|---|---|---|
| `온도/temp_candidate_nx27.csv` | `3cc463977c1cb0f530b097b25d8bdadc973dab36b72e89837218576b5bd4cf76` | 온도 NX 기본 예측(평가 1,440행) | `온도/code/make_submission_v26_temp.py` |
| `EC/output_14/prediction_details.npz` | `57225d8cc1ae8624d30cc34c17e544fc333e09e27ff19c875183c96b954eefca` | EC R3 시드별·TabPFN 문맥별 예측(평가 1,440행) | `EC/model.py` (`EC/reproduce.py`) |
| `EC/hr5_train_base_nosg_oof.csv` | `0e1c626ea65e1a7eb08f5100fb8bac41193efa3c45479ed5c619277e7aa7a4f3` | 학습 기록 360개의 폴드 밖 EC 기본 예측(HR5 학습 입력, 학습 자료만) | `code/retrain/hr5_base_oof/집/클로드/research/make_hr5_base_oof_v1.py` |
| `온도/code/season_ec/training_day_to_season.csv` | `d5d91c6fa0c7a2af912bd64e82b77a602b42638a3c32166cdc5cfc5923433698` | 계절 지표 표(학습 외기만으로 만든 변환) | `EC/model.py` 출력 |
| `온도/code/season_ec/evaluation_day_to_season.csv` | `dde13acf842a85742a30575c722f0f248c9c2f6068bb46e7e65ac8413448bea5` | 위 변환을 평가 기록 번호에 보간한 표(평가 입력 없음) | `EC/model.py` 출력 |
| `EC/st_dong_assign_v1.csv` | `e5c0665bd092413c66137088bf81b9ab2d3f3df66fe60d74045e5af6138c550d` | `EC/c18post.py`를 불러올 때 읽히는 옛 표. 최종 과정의 계산에는 쓰이지 않습니다(그 파일에서 `shrink`만 씀) | — |
