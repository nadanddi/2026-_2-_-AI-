# 재현 패키지 — 팜모니 이미지 미션 제출 01 (팜모니_나단1.zip)

제출 ZIP: `팜모니_나단1.zip` (SHA-256 `4eef08bc12518d67cdeb62138bc6b769b3d207a79c5bcb4d963d149cfc7f3190`)
이 재현 패키지에는 코드·설정·자료 목록(manifest)만 들어 있습니다. AI Hub 원본 사진과 파생 이미지는 재배포 제한으로 넣지 않았습니다.

## 0. 환경
- 자료 추출·프로그램 가짜 제작·저수준 특징: Windows, Python 3.12, `연구실/클로드/local/venv_img`
  - torch 2.5.1 cu121, open_clip_torch 3.3.0, diffusers 0.32.2, transformers 4.46.3, opencv-python 5.0, scikit-learn 1.7.2
- SD/SDXL 생성·CLIP 특징·최종 학습: 연구실 GPU 서버(RTX 6000 Ada), conda 환경 `nadan`(Python 3.10, 같은 패키지 버전), GPU 0번
- 채점 환경 버전 확인: 별도 환경 `nadan_t4`(Python 3.13, torch 2.11, numpy 2.1.3, opencv 5.0.0.93, pillow 11.3)에서 self_check PASS

## 1. 원본 자료 (AI Hub 596 「지능형 수직농장 통합 데이터(딸기)」)
- 내려받은 위치: `01.데이터/1.Training/원천데이터_add_20260807/*.tar`(271개), `01.데이터/2.Validation/원천데이터/*.tar`(36개)
- 추출 순서(사진 목록은 `manifests/im_real_raw_manifest.csv`)
  1. `code/im01_extract_real.py`: VS 36개 × 24장 + TS 60개 × 12장, 개체 라운드로빈
  2. `code/im01b_extract_geumsil.py`: 금실 추가(품종 5:5)
  3. `code/im01d_extract_hard.py`: 흐림·잡음 실제
  4. `code/im01e_extract_farm.py`: 농장·축소 배율 균형
     - 라벨 메타데이터: 팀 내부 정리본 `metadata_all_v1.csv`, `validation_metadata_v1.csv`
- 붉은 조명 원본 제외 목록: `manifests/red_sources.txt` (평균 R−G > 25)

## 2. 학습 자료
| 세트 | 제작 코드 | 목록 |
|---|---|---|
| im_set_v4 (실제 + 프로그램 가짜: 필터 원본 해상도 적용, 작물 분할 합성, 사진풍 스타일) | `im03_make_fakes.py` → `im03b_make_fakes_v2.py` → `im03c_make_fakes_v3.py` → `im03d_composite_v4.py` | `manifests/im_set_v4_manifest.csv` |
| im_set_sd2 (SD1.5 부분 생성·부분 스타일·전체 스타일·전체 생성) | `im07b_sd_fakes_v2.py` (서버 큐 `server/`) | `manifests/im_set_sd2_manifest.csv` |
| im_set_v5 (가짜 제작 5판: 배포 크기 블러, 잎 단위 합성 5b, SDXL 전체 생성·전체 스타일·부분 스타일·부분 생성) | `im15_fakes_v5.py` (서버 `server/s04_gen_v5.sh`) | `manifests/im_set_v5_manifest.csv` |

- 모든 가짜는 행마다 원본(source)·제작 정보(info)를 기록했습니다.
- 난수 시드는 원본 경로의 crc32로 고정했습니다.
- SD/SDXL 가중치(Hugging Face에서 받음): stable-diffusion-v1-5 / -inpainting(fp16), stable-diffusion-xl-base-1.0, sdxl-1.0-inpainting-0.1, sdxl-vae-fp16-fix
  - 다운로드 스크립트: `server/s01_weights.sh`, `server/s02_weights_xl.sh`
- GPU 생성 결과는 같은 시드라도 드라이버·라이브러리 차이로 완전히 같지 않을 수 있습니다.

## 3. 특징과 최종 모델
1. 저수준 38특징: `code/im06_lowlevel.py` (= 제출 `lowfeat.py`)
2. CLIP ViT-L/14@336 특징: `code/im16_backbone_feats.py B2 ...`
   - 336 조각 3×3, 각 L2 정규화 후 평균
3. 지문 DB: `code/im11_fingerprint_db.py` (AI Hub 306 tar 전체, PIL draft 1/8 디코딩 후 12×16 RGB)
4. 최종 학습·내보내기: `code/im20_train_final_img01.py`
   - 로지스틱(C=.05) + HistGradientBoosting(300, .05, 15, l2 1, seed 0)
   - 가중치 = 실제 .5 / 6유형 균등
   - HGB는 numpy 노드 배열로 내보냄(sklearn 결과와 최대차 0)
5. 제출 코드: `submit_code/` (inference.py, vit.py, lowfeat.py, README.md)

## 4. 검증 근거
- 백본 비교: `IM_백본비교_v1.md`, `code/im05k_backbones.py`
  - B2(CLIP L/14@336)가 결합 균형 정확도 .927로 1위였습니다(기존 B1 .896).
- 5판 자료 효과: `code/im05m_eval_v5data.py`
- L2 결합: `IM_후보L1_v1.md`, `code/im05n_eval_B_L2.py`
  - 공개 21장(학습 미사용, 설계 참고) 시드별 21/21/20
  - 우리 실제 오탐: 원본 있음 3.6~4.3%, 원본 없음 6.8~8.1%
- 제출 패키지 검증
  - 패키지 추론 = 학습 경로 예측(최대차 0)
  - self_check PASS
  - 525장 47.7초(GPU + CPU 2코어 제한)
