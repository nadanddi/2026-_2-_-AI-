# 팜모니 이미지 제출 02(팜모니_나단3, C3 유형별 전문가 위원회) 재현 패키지

제출 01 재현 패키지(`제출/이미지01회차_2026-10-12(나단)/팜모니_이미지_재현패키지_01.zip`)에 이어지는 부분만 담았습니다. 기본 판별기는 제출 01 모델 파일 그대로입니다.

## 순서
1. 제출 01 재현(실제 사진 추출, 가짜 제작 v4·SD2·5판, 특징, 지문 DB) → `submit_img01/.../model`
2. GPT 생성·편집 자료(코덱스 GPT 내장 이미지 기능, 프롬프트 전량은 `gpt_records/codex_gpt_v*_manifest.csv`)
   - v1·v2(250장, 학습용): `im21_gpt_set.py`, `im21b_gpt2_set.py` → `im_set_gpt1/2`
   - v3(200장, 시험대 전용): `im21c_gpt3_set.py` → `im_set_gpt3`
   - 원본 목록 `gpt_records/원본*_목록.csv`(AI Hub 원천 파일명)
3. 편집 위치 마스크: `im22_patch_masks.py`(5판 부분 스타일, 생성 시드 재현), `im22b_gpt_masks.py`(GPT, 원본과의 차이)
4. 조각 특징: `im23_patch_feats.py`(ViT-B/16 54조각), 서버 `s14_patch_feats.sh`
5. 검증 흐름(모두 실행 전 등록 문서 `docs/` 있음)
   - G3 → 지름길 반증 → G3c → L2 규칙 개선 → 조합 후보 C
   - 서버 스크립트 `server/s15~s17`
6. 최종 학습·내보내기 `im20c_train_final_img03_C3.py <out_model> <img01_model>` → 점검 `im20d_check_img03.py <패키지>` → 공식 self_check
7. 제출 코드 `submit_code/`

## 환경
서버 conda `nadan`(py3.10, torch, open_clip, scikit-learn), 채점 버전 점검 `nadan_t4`(py3.13, torch 2.11, numpy 2.1.3).
