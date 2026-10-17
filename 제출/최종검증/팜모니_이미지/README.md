# 팜모니_이미지 — 이미지 미션 최종 검증 자료

작성: 연구실 클로드(팀 팜모니), 2026-10-17. 사람이 검토한 뒤 사용자가 접수합니다.

## 0. 한눈에 보기
| 항목 | 내용 |
|---|---|
| 최종 답안 | **팜모니_나단11.zip** |
| 성적 적용 회차 | **R21 (2026-10-16 16:00 회차)**, Accuracy 0.97, 환산점수 25.59 |
| 업로드 시각 | 2026-10-16 15:30경 (사용자 업로드, 플랫폼 기록 기준) |
| 파일 크기 / SHA-256 | 916,144,728 B / `6fe04909341d104dfe7084905f05457b862ceacbd158e645b278520f439b27c1` |
| `submission/` | 운영진 채점 ZIP과 같은 파일이므로 **비워 둠**(위 버전 정보로 대신함). 원본 위치: `연구실/클로드/local/submissions_img/팜모니_나단11.zip`, 구글 드라이브 `farmai_t4/팜모니_나단11.zip` |
| 구성 요약 | 나단9(기본 판별기 + 원본 대조 규칙 + 유형별 전문가) + **GPT 편집 전문가 E_gpt**(형식 게이트 안, 진짜 확정이 아닌 사진에만) |

### 폴더 구성
| 폴더 | 내용 |
|---|---|
| `README.md` | 이 문서(①~⑧) |
| `submission/` | 비움(채점 ZIP과 같음) |
| `code/submission_inference/` | 최종 ZIP 안의 추론 코드 그대로(`inference.py`, `lowfeat.py`, `vit.py`, `requirements.txt`, `README.md`, `config.json`) |
| `code/claude_image/` | 학습 자료 추출·가짜 제작·특징·학습·검증·조립 스크립트 전체(`im01`~`im74`)와 실행 전 등록·결과 문서(`IM_*.md`), 회차별 제출 코드(`submit_img01`~`09`) |
| `code/history_repro/` | 회차별 재현 패키지 `_01`·`_02`·`_03`·`_04_04b`(04·04b는 원래 낱개 파일 묶음이라 한 폴더로 모음). 서버 실행 스크립트 `server/s*.sh`, Colab 정밀 지문 스크립트 포함 |
| `code/codex_E_gpt/` | E_gpt 학습·검증 코드와 결과(연구실 코덱스 작성: `run_v3.py`, `verify_v*.py`, `결과_v1.md` 등) |
| `models/` | 최종 ZIP의 `model/` 폴더 전체(사전학습 가중치 포함). 해시: `data/models_sha256.txt` |
| `data/manifests/` | 학습·검증 이미지 목록(파일명·라벨·유형·제작 정보·원본 파일명·촬영 묶음). `server/`는 GPU 서버, `local_pc/`는 연구실 PC에서 만든 세트 |
| `data/gpt_generation/` | GPT 생성·편집 자료의 요청서·지시문·기록·검증 파일(이미지 제외) |
| `data/README_data.md` | 자료 출처·이용조건·접근 절차·선택 기준·공개 샘플 사용 내역 |
| `logs/` | GPU 서버 실행 로그 108개(`logs/logs/`)와 재현·시간 측정 기록(`logs/README_logs.md`) |

> 용량 때문에 git 저장소의 `제출/최종검증/팜모니_이미지/`에는 `models/`가 없습니다. 접수용 전체 ZIP은 `연구실/클로드/local/최종검증/팜모니_이미지.zip`(models 포함)입니다.

---

## ① 사용 자료와 출처
1. **AI Hub 「지능형 수직농장 통합 데이터(딸기)」(데이터셋 596)**: 실제 사진, 가짜 제작의 원본, 지문 DB의 원천입니다. 이용 신청·승인 후 팀원 계정으로 원천데이터 tar 306개(Training `원천데이터_add_20260807` 271, Validation `원천데이터` 35, 2026-10-17 드라이브에서 다시 셈)를 내려받았습니다. 재배포 제한 때문에 원본·파생 이미지는 넣지 않았습니다. 지문 DB(사진별 숫자 통계)는 10-12 팀 결정에 따라 출처를 밝히고 포함했습니다. 자세한 내용은 `data/README_data.md`에 있습니다.
2. **자체 제작 가짜**: 위 실제 사진에 문제설명서의 6유형(필터·합성·부분 생성·부분 스타일·전체 생성·전체 스타일)을 적용해 만들었습니다.
   - 프로그램 처리: 필터, 합성, 색·톤 스타일
   - Stable Diffusion 1.5 / 1.5 inpainting, SDXL base 1.0 / SDXL inpainting 0.1, sdxl-vae-fp16-fix
   - Qwen-Image-Edit(Apache-2.0, Lightning 8단계 LoRA)
3. **GPT 생성·편집 자료**: 연구실 코덱스가 ChatGPT 내장 이미지 생성·편집 기능으로 만들었습니다(AI Hub 원본에 편집 지시). 묶음별 수량은 다음과 같습니다.

   | 묶음 | 수량 | 내용 |
   |---|---|---|
   | gpt1 | 100 | 전체 생성·전체 스타일·부분 생성·부분 스타일 각 25 |
   | gpt2 | 150 | 부분 생성 100, 부분 스타일 50 |
   | gpt3 | 200 | 부분 생성 100, 부분 스타일 50, 전체 다시 그리기 50 |
   | gptr | 150 | 같은 구도 전체 다시 그리기 |

   요청서·지시문·기록은 `data/gpt_generation/`에 있습니다.
4. **사전학습 가중치**: OpenAI CLIP ViT-L/14@336px, ViT-B/16(MIT). open_clip 3.3.0 `pretrained="openai"`로 받은 시각부만 fp16 safetensors로 저장했고, 추가 학습은 하지 않았습니다. 생성 모델(SD·SDXL·Qwen)은 학습 자료 제작에만 썼고 제출물에는 넣지 않았습니다.
5. **공개 샘플 21장**: 학습에 넣지 않았습니다. 다만 ① 파일 형식 실측과 원본 대조 규칙 착안(설계 참고), ② 10-10~10-12 초기 후보 구성(가짜 제작 5판 포함 여부, 기본 판별기 조합)을 고를 때 **검증 지표**로 썼습니다(계획한 열람 상한을 넘어 5회 이상), ③ 이후 모든 후보의 '21/21 유지' 비열화 점검에 썼습니다. 자세한 내용은 `data/README_data.md` 4절에 있습니다.
6. 그 밖의 외부 자료는 없습니다. 이전 대회 자료도 쓰지 않았습니다.

## ② 특징 구성과 사용하는 시간 범위
- 이미지 미션이라 시간 축이 없습니다. **각 이미지는 그 이미지 파일 하나만 입력으로** 계산합니다.
- 특징
  - 의미 특징: CLIP ViT-L/14@336, 원해상도 336 조각 3×3 평균
  - 국소 조각 특징: ViT-B/16, 224 조각 54개(간격 112)
  - 저수준 38특징: 잡음 잔차·선명도·JPEG 격자·채도·주파수·타일 편차(`lowfeat.py`)
  - 원본 대조 특징: 12×16 색 지문, 정밀 지문(흑백 24×32 + 12×16 라플라시안 에너지)과 학습 전에 만든 AI Hub 지문 DB의 L1 거리
  - 파일 형식 판별: JPEG·4:2:0·Q92 양자화표

## ③ 평가 입력의 사용 범위
- 평가 이미지끼리 정보를 주고받지 않습니다. 이미지마다 독립적으로 계산합니다(다른 이미지의 값·통계를 쓰지 않음). 경로를 정렬한 뒤 8장씩 묶어 GPU로 계산하므로 **입력 순서와는 무관**합니다. 다만 묶음 구성(함께 들어온 파일 집합)이나 GPU fp16/CPU fp32 차이로 확률에 작은 수치 차가 생길 수 있습니다. 나단3·4 사이 최대 .008, 판정 1장 차이가 있었습니다.
- 공식 `self_check.py` 순서 독립성 검사는 OK였습니다(10-16, 서버, `logs/README_logs.md`).
- **비공개 평가 자료는 학습·설정 선택에 쓰지 않았습니다.** 운영진 시험 이미지는 코드 제출 방식이라 팀이 접근할 수 없었습니다. 정답을 보지 않는 시험 입력 통계 집계(내부 명칭 E4)도 10-16에 제안됐지만 규정 취지상 하지 않기로 했습니다(`code/claude_image/` 작업 기록, `제출/제출기록.md`).
- Colab 시간 측정에 쓴 `test_images_521.zip`은 **우리 자료로 구성한 시간 측정용 대용품**입니다. 이름의 521은 시험 장수에 맞춘 것일 뿐 운영진 시험 자료가 아닙니다. 구성은 gpt3 200 + 처리 실제(realn) 300 + 공개 샘플 21이고, 목록과 SHA는 `logs/test_images_521_구성.txt`에 있습니다.
- 리더보드 점수 사용 내역은 ⑤ 끝에 정리했습니다.

## ④ 모델과 학습 자료 범위
| 구성 요소 | 학습 자료(장수) | 만든 스크립트 |
|---|---|---|
| 기본 판별기: CLIP 로지스틱 `b2_lr.npz` + 저수준 부스팅 `low_hgb.npz` | AI Hub 실제 2,467 + 자체 가짜 7,124(v4 프로그램 가짜·SD1.5 sd2·가짜 제작 5판 SDXL). 가중 = 실제 .5 / 6유형 균등 | `im20_train_final_img01.py`(재현 패키지 01) |
| 원본 대조 DB `fp_db.npz.xz` | AI Hub 원천 152,043장의 12×16 색 평균(학습 없음) | `im11_fingerprint_db.py` |
| 정밀 지문 DB `finefp_db.npz.xz`와 문턱(config `finefp`) | 같은 152,043장의 흑백 24×32 + 라플라시안 에너지 / 문턱은 처리 흔들기 실제 1,500장 분위 | Colab `colab_정밀지문.py`·`_v2.py`(재현 03), 문턱 `im36b_finefp_eval_v2.py` |
| 전문가 E_pg·E_ps `e_pg_lr.npz`·`e_ps_lr.npz`, 국소 조각 E_L `e_l_lr.npz` | 실제 + 해당 유형 가짜(SD 계열 + GPT gpt1·gpt2 250 + 새 실제 795 + 되붙임 7판 234 + Qwen 390). 총 11,260행 | `im20c`·`im20e_train_final_img05_C3v2.py`, 조각 마스크 `im22`·`im22b`, 조각 특징 `im23` |
| 전문가 H(E_f·E_ps2·E_g7) `h_experts.npz` | 새 AI Hub 실제 1,800(묶음 40) + 필터 906 + 부분 스타일 501 + 국소 재생성 393 + 처리 흔들기 실제 1,800 | `im41`(추출)·`im42`(제작)·`im45`(흔들기)·`im43`(특징)·`im44 no .0025 .001 h2 jig`(학습)·`im47`(조립) |
| GPT 재그리기 전문가 E_gr `gr_expert.npz` | 양성 gptr 학습 절반 72, 음성 h3 실제 1,800 + 흔들기 1,800 + 짝 실제 72 | `im58_gptr_set.py`, `im59_ip_train_eval.py`, `im61_build_img09.py` |
| **GPT 편집 전문가 E_gpt** `gpt_expert.npz`, config `gpt` | 양성 GPT 4묶음 600(gpt1 100·gpt2 150·gpt3 200·gptr 150), 음성 h3 실제 1,800 + 흔들기 1,800 + gptr 짝 실제 150 = 3,750 | 코덱스 `code/codex_E_gpt/run_v3.py`(학습·θ), 조립 `im74_build_img11.py` |

- 모든 로지스틱은 입력 표준화 후 학습했습니다(class_weight balanced 또는 표본 가중, 각 스크립트 참고).
- 원본 촬영 묶음(group)이나 농장 단위로 교차검증해, 같은 원본의 실제와 가짜가 다른 폴드에 나뉘지 않게 했습니다.

## ⑤ 후처리 (값·계산식·결정 근거)
최종 확률은 이미지 하나마다 다음 순서로 계산합니다(`inference.py` `_final`). 판정 임계값은 0.5(운영 기준 그대로)이고, 따로 확률 보정은 하지 않았습니다.

| 순서 | 계산 | 값 | 결정 근거 |
|---|---|---|---|
| 1 | p = (p_CLIP + p_low) / 2 | 0.5/0.5 고정 | 교차검증에서 단일 갈래보다 균형 정확도가 높았음(재현 01) |
| 2 | 원본 대조 R1: 3.5 < d1 < 20, gap > 8, p ≥ .25 → p ← max(p, .9) | 3.5·20·8·.25·.9 | 학습 자료 분포. 원본 있는 실제 d1 99분위 3.29(축소·재확대 대조군 3.72), 이웃 프레임만 있는 실제 gap 95분위 8.06 |
| 3 | 전문가 E_pg·E_ps·E_L: p ← max(p, g_k(s_k)), g(s) = .5s/θ (s<θ), .5+.5(s−θ)/(1−θ) | θ = .7262 / .8969 / .99908 | 학습 OOF에서 기본이 실제로 본 실제의 0.4%만 뒤집는 값을 시드 6·7·8 × 5폴드 교차 적합으로 구해 평균 |
| 4 | 정밀 원본 대조: dg1 ≤ 1.011 & gap ≥ 3.474일 때, 국소 차이 lg > 6.569 또는 le > 52.95 → p ← max(p, .95)(가짜 확정). lg ≤ 3.933 & le ≤ 24.24 & \|ge\| ≤ 13 & p < .8 → p ← min(p, .3)(진짜 확정, rc) | config `finefp` | 처리 흔들기 실제 1,500장 분위(원본 판정 99.5%, 가짜 확정 99.9%, 진짜 확정 95%) |
| 5 | 형식 게이트: JPEG·4:2:0·PIL Q92 양자화표일 때만 6·7 적용 | — | 전문가가 배포 형식으로만 학습됨. 공개 21장 실측 형식과 같음 |
| 6 | 전문가 H·E_gr: p ← max(p, g(s_k, θ_k)), rc이면 θ_rc 사용 | θ: E_f .6151, E_ps2 .99960, E_g7 .99982, E_gr .03107 / θ_rc: .8481, .99976, .99991, .08777 | 실제 풀(학습 실제 묶음 5폴드 OOF + 미사용 실제 1,395)에서 실제 0.25%만 뒤집는 분위(rc는 0.1%) |
| 7 | **E_gpt**: 게이트 안 & rc 아님 → p ← max(p, g(s_gpt, θ_gpt)) | θ_gpt = **.907316** | 실제 풀 3,345장의 0.25% 분위. 학습 스크립트가 θ_rc(.9630)도 계산했지만 rc 사진에는 E_gpt를 아예 쓰지 않으므로 사용하지 않음. rc 제외는 아래 설명 |

**정의**
- 정밀 지문: 사진을 덮게 축소·가운데 768×1024 크롭 후 흑백 24×32 평균 g(768값)과 64px 구역 12×16 라플라시안 분산 log값 e(192값). DB 1위와의 거리 dg1 = 평균 abs(g−g_db), gap = 2위 거리 − dg1.
- 국소·전체 차이: dg = abs(g − g_db1), de = e − e_db1. **lg** = (dg − 중앙값(dg)) 상위 2% 평균, **le** = abs(de − 중앙값(de)) 상위 5% 평균, **ge** = 중앙값(de).
- 12×16 색 지문(R1): 사진 12×16 RGB 평균과 DB의 평균 절대차 1위 d1, gap = 2위 − 1위.
- 형식 게이트: PIL로 연 파일이 JPEG이고, 크로마 서브샘플링이 4:2:0(`JpegImagePlugin.get_sampling == 2`)이며, 양자화표가 PIL quality=92로 저장했을 때의 표와 **완전히 같을 때**만 참.
- 예외 대체(이미지별 고정)
  - 저수준 특징 실패 → p = p_CLIP만 쓰고, 저수준을 입력으로 쓰는 E_f·E_gr·E_gpt는 미적용
  - CLIP 실패 → p_low만
  - 둘 다 실패 → .5
  - 원본 대조 실패 → R1·정밀 대조 규칙 미적용
  - 전문가 계산 실패 → 그 전문가 미적용
  - NaN → .5

**E_gpt의 rc 제외 결정 (사후 변형임을 밝힘)**
- E_gpt 첫 판(rc에도 적용)은 GPT 묶음 하나씩 빼기 검증에서 효과가 있었습니다. 나단9가 놓친 26장 중 14장을 새로 잡았습니다.
- 하지만 실제 비용 기준을 근소하게 넘어 불통과였습니다(합집합 실제 오탐 +0.12%p, R5 균형 정확도 −0.28%p).
- 분해해 보니 새 실제 오판 대부분이 진짜 확정(rc) 사진이었고, GPT 가짜 600장 중 rc는 0장이었습니다. 그래서 rc 사진은 제외했습니다.
- 같은 자료로 다시 집계하면 사전 기준을 충족합니다: 합집합 +0.03%p, R5 변화 0, R2 −0.027%p(실제 1장), 빼기 검증 14장 유지. 다만 **불통과를 본 뒤 같은 자료로 만든 사후 변형**이라 수치가 낙관적일 수 있습니다.
- 이 결정은 내부 자료로만 했고, 리더보드는 쓰지 않았습니다. 근거: `code/claude_image/IM_Egpt_rc제외_v1.md`.

**리더보드 점수 사용 내역(사실대로)**
- 계수·문턱·가중은 리더보드 점수로 고르지 않았습니다.
- 회차마다 미리 정한 하나의 가설 후보를 내고, 점수로 채택 여부만 비교했습니다. 공개 점수로 제출끼리 비교하는 것은 운영위원회가 2026-10-05 중간 안내 뒤 추가 공지(팀 기록: 카탈로그 6.344, 정형 미션 질의 답변)에서 허용한다고 안내한 범위로 이해했습니다. 이미지 미션에 같은 기준이 적용된다고 본 것은 팀의 해석입니다.
- **최종 답안 선택**
  - 사전에 고정한 규칙은 "LB Accuracy ≥ .98이면 후보 유지, 아니면 나단9로 되돌림"이었습니다.
  - R21에서 나단11은 Accuracy 0.97로 표시가 나단9와 같았습니다. 하지만 **공식 환산점수가 23.82 → 25.59로 올라** 나단11을 최종으로 유지했습니다.
  - 즉 최종 선택에는 **공개 리더보드의 환산점수 비교를 사용**했습니다. 모델 값은 바꾸지 않았습니다.

## ⑥ 실행 환경·설치·실행 순서·장비·예상 시간
- **추론(채점 환경)**
  - `requirements.txt`에는 추가 허용 패키지 2개만 고정했습니다: opencv-python==5.0.0.93, safetensors==0.8.0
  - torch·numpy·pillow는 채점 기본 환경을 그대로 씁니다. 같은 버전으로 점검한 환경은 conda `nadan_t4`이고 목록은 `logs/nadan_pipfreeze_t4.txt`에 있습니다(Python 3.13, torch 2.11, numpy 2.1.3, pillow 11.3).
  - 네트워크·다운로드 없음. GPU면 fp16, 없으면 CPU fp32
  - 실행: ZIP을 풀고 `inference.load_model("<팀폴더>/model")` → `inference.predict(model, image_paths)`
- **실측**
  - Colab T4, `colab_시간측정.py`, 524장 환산 합계: 265.1 / 273.7 / 272.5 / 272.5초(한도 300초). 사용자가 10-16에 측정한 수치이며, Colab 화면 원본 출력은 보관하지 않았습니다.
  - 서버(RTX 6000 Ada) 공식 self_check: PASS, 공개 21장 load 7.3초 + 예측 1.9초, 순서 독립 OK
- **학습 환경**
  - 연구실 PC(Windows, Python 3.12, `venv_img`): AI Hub tar 추출, 프로그램 가짜 제작, 저수준 특징
  - 연구실 GPU 서버(ADILAB, RTX 6000 Ada 48GB, GPU 0만 사용, conda `nadan`, Python 3.10): SD·SDXL·Qwen 생성, CLIP 특징, 학습. 실제 설치 버전은 `logs/nadan_pipfreeze_nadan.txt`(pip freeze)
  - 지문 DB: Google Colab(`colab_정밀지문.py`)
- **사전학습 가중치(제출 모델에 포함된 CLIP)**
  - ViT-L/14@336 openai: open_clip 3.3.0 `ViT-L-14-336`, pretrained=`openai`, quick_gelu
    - 원 체크포인트: https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
    - 경로의 긴 문자열이 OpenAI가 표기한 SHA-256입니다.
    - 실제로 받은 사본: HF `timm/vit_large_patch14_clip_336.openai` blob `fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0`
  - ViT-B/16 openai
    - 원 체크포인트: https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
    - HF 사본: `timm/vit_base_patch16_clip_224.openai` blob `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`
  - 변환 절차: open_clip으로 모델을 불러 시각부(`model.visual`) state_dict만 fp16 safetensors로 저장했습니다(`im20_train_final_img01.py` 56~59행 = ViT-L, `im20c_train_final_img03_C3.py` 87~89행 = ViT-B). `vit.py`로 다시 읽은 출력이 open_clip과 같음(최대 차 0)을 확인했습니다.
  - 서버의 가중치 파일 목록·크기: `logs/server_weights_files.txt`
- **생성 모델 가중치(학습 자료 제작에만 사용, 제출물에 없음)**
  - HF에서 받은 폴더 스냅숏을 썼습니다(SD1.5 inpainting, SD1.5, SDXL base 1.0, SDXL inpainting 0.1, sdxl-vae-fp16-fix, Qwen-Image-Edit, Qwen Lightning LoRA).
  - 다운로드 스크립트(`server/s01_weights.sh`, `s02_weights_xl.sh`)에 revision을 고정하지 않았습니다. 확인된 커밋은 stable-diffusion-v1-5 `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` 하나뿐입니다.
  - 그래서 같은 이름으로 다시 받으면 버전이 다를 수 있습니다. 파일 크기 목록은 `logs/server_weights_files.txt`에 있습니다.
- **전체 재학습 순서**(각 단계의 입력·출력 경로는 스크립트 머리말에 있음)

  | 단계 | 명령(대표) | 산출 | 대략 시간 |
  |---|---|---|---|
  | 1. AI Hub 실제 추출 | `im01_extract_real.py` → `im01b`·`im01d`·`im01e` | `local/im_real_raw/` | tar 순차 읽기 수 시간 |
  | 2. 가짜 제작 v4·sd2·5판 | `im03`→`im03b`→`im03c`→`im03d`, `im07b_sd_fakes_v2.py`, `im15_fakes_v5.py`(서버 `s04_gen_v5.sh`) | `im_set_v4`·`sd2`·`v5` | GPU 수 시간 |
  | 3. 특징·지문 DB | `im06_lowlevel.py`, `im16_backbone_feats.py B2`, `im11_fingerprint_db.py` | 특징 npz, `fp_db.npz` | 수십 분 / DB 수 시간 |
  | 4. 기본 판별기 | `im20_train_final_img01.py` | `b2_lr.npz`, `low_hgb.npz`, `b2_visual_fp16` | 수 분 |
  | 5. GPT 자료·전문가 C3 | 코덱스 GPT 생성(`data/gpt_generation`) → `im21_gpt_set.py`·`im21b`·`im21c` → `im22`·`im22b` → `im23` → `im20c` | `e_pg/e_ps/e_l_lr.npz` | 수십 분 |
  | 6. C3v2 | `im30_extract_new.py` → `im31_qwen_edit.py`·`im15d_pgen_v7.py` → `im34b` → `im20e_train_final_img05_C3v2.py` | 전문가 재학습 | GPU 수 시간 |
  | 7. 정밀 지문 | Colab `colab_정밀지문.py`(+`_v2`) → `im36b_finefp_eval_v2.py`(문턱) | `finefp_db.npz`, config `finefp` | Colab 수 시간 |
  | 8. 전문가 H | `im41_extract_h.py` → `im42_make_h.py cpu/gpu/merge` → `im45_jig_h3.py` → `im43_h_feats.py im_set_h3 im_set_h3j ...` → `im44_h_train_eval.py no .0025 .001 h2 jig` → `im47_build_img08.py` | `h_experts.npz` | GPU 수 시간 |
  | 9. E_gr | `im58_gptr_set.py` → `im43_h_feats.py im_set_gptr im_set_gptr_src` → `im59_ip_train_eval.py` → `im61_build_img09.py` | `gr_expert.npz` | 수십 분 |
  | 10. E_gpt | `im43_h_feats.py im_set_gpt2`(특징이 없을 때) → 코덱스 `code/codex_E_gpt/run_v3.py`(입력: 서버 `data/local/feats_h/*.npz`, 출력 `final_model.npz`·`final_theta.json`) | E_gpt 가중치·θ | 수 분 |
  | 11. 조립 | `im74_build_img11.py`(입력: 나단9 ZIP, `im74_inputs/`의 세 파일) | `팜모니_나단11.zip` | 수 분 |

  - 단계 1~7은 재현 패키지 `_01`·`_02`·`_03`의 `README_재현.md`, 8~9는 `_04_04b`에 더 자세히 있습니다.
- **난수**: 가짜 제작 시드는 원본 경로 crc32로 고정했습니다. GPU 생성물은 드라이버·라이브러리 차이로 완전히 같지 않을 수 있습니다. 학습은 random_state를 고정했습니다.

## ⑦ 대응 정보
- 성적 적용 회차: **R21 (2026-10-16 16:00)**
- ZIP 파일명: **팜모니_나단11.zip**
- 업로드 시각: 2026-10-16 15:30경
- SHA-256: `6fe04909341d104dfe7084905f05457b862ceacbd158e645b278520f439b27c1`

## ⑧ 최종 답안과 코드의 대응
| ZIP 안 파일 | 만든 코드 | 비고 |
|---|---|---|
| `inference.py` | 나단9 `submit_img09/inference.py`에 E_gpt 두 줄 블록 추가(`im74_build_img11.py`) | 원본과의 차이는 `code/claude_image/im74_build_img11.py`의 치환 문자열 |
| `lowfeat.py`, `vit.py` | 제출 01부터 같음(`im06_lowlevel.py` 계산과 동일, `vit.py`는 open_clip과 출력 차 0) | `submit_img01`·`09`의 사본과 줄바꿈(CRLF/LF)만 다름 |
| `model/b2_visual_fp16.safetensors`, `b16_visual_fp16.safetensors` | open_clip `ViT-L-14-336`·`ViT-B-16`(openai) 시각부 fp16 저장(`im20_train_final_img01.py` 56~59행, `im20c_train_final_img03_C3.py` 87~89행) | 사전학습, 추가 학습 없음 |
| `model/b2_lr.npz`, `low_hgb.npz` | `im20_train_final_img01.py` | 제출 01 그대로 |
| `model/fp_db.npz.xz` | `im11_fingerprint_db.py` → 나단7에서 lzma 재압축 | 값 동일 |
| `model/finefp_db.npz.xz` | `colab_정밀지문.py`(+`_v2`) → 나단6 조립 → 나단7 lzma 재압축 | |
| `model/e_pg_lr.npz`, `e_ps_lr.npz`, `e_l_lr.npz` | `im20e_train_final_img05_C3v2.py` | 나단5부터 그대로 |
| `model/h_experts.npz` | `im44_h_train_eval.py`(tag h2, jig) → `im47_build_img08.py` | 나단8부터 |
| `model/gr_expert.npz` | `im59_ip_train_eval.py` → `im61_build_img09.py` | 나단9부터 |
| `model/gpt_expert.npz` | 코덱스 `run_v3.py` 최종 학습(`final_model.npz`) → `im74_build_img11.py`에서 키 이름만 맞춰 저장 | 나단11 |
| `model/config.json` | 각 빌드 스크립트가 값 기록. 나단11은 `gpt.theta`(.9073158431053205, 코덱스 `final_theta.json`)와 version 추가 | |
- `im74_build_img11.py`는 같은 폴더의 `readme11.md`를 읽고, 코덱스 산출물 경로(`연구실/코덱스/local/im_egpt_20261016_v1/corrected_v3/`)를 씁니다. 이 패키지에서는 세 입력 파일을 `code/claude_image/im74_inputs/`에 넣었으니, 다시 실행할 때 경로만 맞추면 됩니다.

## 이전 제출 자료 보관 (무효 시 복귀 후보용)
`data/README_data.md` 5절에 회차별 ZIP·코드·모델 위치와 해시를 정리했습니다.
