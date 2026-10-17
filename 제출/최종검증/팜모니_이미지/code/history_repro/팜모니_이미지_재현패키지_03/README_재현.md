# 팜모니 이미지 제출 03(팜모니_나단6) 재현 패키지
제출 01·02 재현 패키지에 이어지는 부분만 담았습니다.
1. 새 AI Hub 추출 `im30_extract_new.py`(구글 드라이브 tar, 순차 읽기) → 생성 `im31_qwen_edit.py`(Qwen-Image-Edit, Apache-2.0, Lightning 8단계 LoRA)·`im15d_pgen_v7.py`(되붙임 국소 편집 7판) — 서버 `s18`·`s19`
2. 봉인 확증 `im32`·`im33`(문서 `IM_봉인확증_C3_*`)
3. C3v2: `im34b`(학습 묶음) → `im20e`(전문가 재학습·θ 교차적합) → `im35b`(판정) — 서버 `s20b`
4. 정밀 지문: Colab `colab/colab_정밀지문.py`(+ `_v2` 역순) → 판정 `im36b`(처리 흔들기 진짜로 문턱) → 패키지 조립(정밀 DB 합치기 + config의 finefp 문턱)
5. 제출 코드 `submit_code/`, Colab T4 시간 측정 `colab/colab_시간측정.py`
