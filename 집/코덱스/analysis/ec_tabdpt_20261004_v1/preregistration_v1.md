# family20 · TabDPT CPU 단일 부품 교체 · 실행 전 고정 · 2026-10-04

기존 EC 계절 v2의 .2 PFN raw만 TabDPT v1.3.0 mean으로 바꾼다. .8 R3와 동일 FULL38·계절·22fold/3seed·shrink/학습 범위 clip은 유지한다. 기존 공개 검증을 반복 사용한 새 후보이며 독립 홀드아웃 결과로 해석하지 않는다. 운영 특징 DP1/DP2와 결합하지 않는다.

- 공식 source commit 97e5494431e9527c7edb31cb4dcfc5f00b232fdf.
- 공개 weight revision a5ca6e01c0fa09ec68c73e958e5199d1932abb3a, tabdpt1_3.safetensors 252233296 bytes/SHA 97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb. 공식 API의 gated=false/private=false·해당 revision Apache2 및 익명 HEAD200 확인 후 익명 취득/로컬 hash 검산 PASS. 계정/동의/업로드0.
- own-local overlay faiss-cpu1.12.0/OmegaConf2.3.0/antlr4-python3-runtime4.9.3/Hub0.36.0/고정 TabDPT source. 공용 환경 수정0. 설치 report/archive hashes와 실제 runtime_probe_v3 검사 결과를 보존한다. 최초 sandbox METADATA 접근 실패는 버전/문턱을 우회하지 않고 읽기 권한을 확인해 분리 검산한다.
- `normalizer=standard, missing_indicators=False, clip_sigma=8, feature_reduction=pca, context_reduction=retrieval, faiss_metric=l2, device=cpu, use_flash=False, compile=False`.
- predict `context_size=512, n_ensembles=8, batch_size=8, output_type=mean, seed=7/101/2024`. torch/FAISS/BLAS thread1, deterministic algorithms. n_train>512/max_features≥38 필요, 설정 변경 없이 중단한다.
- candidate = clip(shrink(.8 R3raw+.2 TabDPTraw),train min,max). baseline의 동일 원 raw 부품/row ID/공개 labels/bounds/전체 22fold 재검산 후 적합한다.
- 첫 fold raw 반복/단독/역순/타 query 변경/fresh fit 절대차≤1e−6, 같은 과거 prefix를 보존한 최종 절대차≤2e−7. manual shrink와 원 입력 미래/다른 농장 변조의 특징 인과성도 검사한다. score를 보기 전에 첫 audit PASS를 요구한다. 설정이나 문턱을 사후 완화하지 않는다.
- 모든 seed×DIAG10/A/B/EXT10/EXT12 엄격 개선 및 DIAG p_worse<.025/20=.00125·조정 CI 상한<0. 고EC31일/일반329일·기존 후기179 진단은 설명용이며 gate/비중 재선정0.
- source/입력/ordered ID/배열/범위/기초cache/환경/weight 해시를 cell signature에 기록한다. 부분 artifact나 재개 signature 불일치 시 적합 전에 중단하며 보존한다. 완료 experiment를 재학습하지 않는다.
- 원시 EC 정답/잠금/EL1재채점/test예측/제출물0. 공개 계절v2 OOF에서 허용된 360일 labels만 쓴다. 사용자의 새 지시에 따라 실험 완료 때마다 클로드 최신 Git/일지/로그/결과를 먼저 갱신한다. 현재 DB1/DP1 완료기각 확인, DP2 live이므로 중복 실행0.

실행 소스와 preflight/runtime 결과를 main에 먼저 커밋한 뒤 run_v1.py를 실행한다. 이 사전서 작성 시점 모델 fit/predict0이며 설치·가중치 취득을 모델 학습으로 세지 않는다.
