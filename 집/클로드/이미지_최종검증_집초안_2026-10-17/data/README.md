# data — 학습 이미지 목록과 자료 출처

## 1. 재배포 제한 자료: AI Hub 원천 사진
- **자료:** AI Hub 「지능형 수직농장 통합 데이터(딸기)」 데이터셋 번호 596
  - 과학기술정보통신부 재원, 한국지능정보사회진흥원(NIA) 지원으로 구축됐습니다.
- **이용조건:** AI Hub 이용 신청·승인 후 연구·AI 학습 목적으로 사용합니다. 원본 재배포는 하지 않습니다. 그래서 원본 사진과 그 파생 이미지(자체 제작 가짜)는 이 폴더에 넣지 않았습니다.
- **접근 절차:**
  1. AI Hub(aihub.or.kr)에서 위 데이터셋을 이용 신청하고 승인받습니다.
  2. 원천데이터 tar를 받습니다. 쓴 것은 Training `원천데이터_add_20260807` 271개, Validation `원천데이터` 36개입니다.
- **선택 기준:**
  - tar마다 개체를 돌아가며 고르게 추출했습니다. 품종·농장·축소 배율 구간이 고르게 되도록 추가 추출했습니다.
  - 붉은 LED 조명 사진(평균 R−G > 25)은 뺐습니다.
  - 배포 형식에 맞춰 처리했습니다: 덮게 축소 → 가운데 768×1024 크롭 → JPEG 품질 92.
  - 추출 코드(`code/training_연구실클로드/`): `im01_extract_real.py`, `im01b`·`im01d`·`im01e`(추가 추출), `im30_extract_new.py`, `im41_extract_h.py`, `im51_extract_i.py`, `im63_extract_r5.py`.
  - 가짜 제작 코드: `im03*`, `im07*`, `im15*`, `im42`(H 자료) 등.
- **재현:** 아래 목록의 `source` 열(tar 이름\파일 이름)로 같은 원본을 찾을 수 있습니다. 가짜는 목록의 `kind`·`info` 열과 제작 코드로 다시 만듭니다.

## 2. 학습 이미지 목록 (`manifests/`)
- `lab_claude_local/im_set_*/manifest.csv`: 이미지 묶음별 목록입니다.
  - `file`: 파일명
  - `label`: 0 실제, 1 가짜
  - `kind`: 유형(real, filter, composite, full_style, partial_style, partial_gen, full_gen 등)
  - `info`: 제작 방법·변수
  - `source`: AI Hub 원본 tar\파일
  - `group`, `farm_code`: 촬영 묶음·농장(교차검증 폴드 기준)
  - `other`: 합성에 쓴 다른 원본
- `lab_claude_local/im_*_raw/manifest*.csv`: AI Hub에서 추출한 원본 목록입니다(tar, split, file, group, farm_code, size).
- `lab_codex_local/ai_gen_work_*`, `gpt_regen_20261015/`: GPT 생성·편집 기록입니다.
  - 각 결과의 SHA256, 원본과 원본 SHA256, AI Hub 원본 경로, 요청 문장(prompt), 검증 결과가 들어 있습니다.
- `lab_claude_local/match/lookup_public.csv`: 공개 샘플 21장과 AI Hub 원본의 대조 결과입니다(설계 참고 분석, 학습 미사용).

**묶음별 용도 (README ④ 참고)**

| 묶음 | 용도 |
|---|---|
| `im_set_v4`, `im_set_sd2`, (`im_set_v5`) | 기본 판별기 학습 |
| `im_set_gpt1`, `im_set_gpt2`, (`im_set_newreal`, `im_set_v7tr`, `im_set_qwentr`) | 전문가 E_pg·E_ps·E_L 학습 |
| (`im_set_h3`, `im_set_h3j`) | 전문가 H2 학습. `im_set_h1`은 그 전 판(H v1, 기각) |
| (`im_set_gptr`, `im_set_gptr_src`) | E_gr 학습 |
| `im_set_gpt1·gpt2·gpt3`, (`gptr`) | E_gpt 학습 |
| `im_set_r2`, (`im_set_r3`, `im_set_r4`), `im_r5_raw` | 봉인 검증(학습 미사용) |
| `im_set_realg`, `im_set_realn` | 실제 풀(문턱 θ 분위 계산) |
| `im_set_v1`~`v3`, `im_set_sd3` | 이전 판 제작 자료(실험·비교) |

- 괄호 안 묶음은 목록이 연구실 GPU 서버에만 있어 아직 넣지 못했습니다(README "추가 접수 예정").
- 용도는 학습 스크립트에서 읽는 묶음 이름으로 확인했습니다. 묶음별 정확한 장수는 서버 목록으로 최종 확인이 필요합니다.
