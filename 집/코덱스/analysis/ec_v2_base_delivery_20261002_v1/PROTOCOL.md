# K1 기존 EC v2 전달본 사전등록 · 2026-10-02 집 코덱스

사용자가 직접 지정한 EC개선2단계 분담 요청서의 K1에 따라 EC 전용 row_id/sub_ec CSV, 재현 ZIP, 한국어 설명을 만든다. 실제 플랫폼 제출·온도 모델 생성은 수행하지 않는다. 기존 EC v2를 전달하며 새 개선/확정 제출 구성으로 부르지 않는다.

모델은 기존 원시 R3(.6ET+.3LGB+.1MLP)의 시드7/101/2024 평균80% + TabPFN V2의 context1..4 평균20%, 원시예측 혼합 뒤 .5현재+.5당일 현재까지 평균과 학습범위clip. STATE 정답특징/보정은0. ET/PFN FULL38, LGB/MLP BASE14. CPUfloat32, checkpoint 고정, TabPFN 각context2000행/내부4estimators, 모델/BLAS4thread, NumPy2.5.3 및 기존 버전고정.

최종잠금40일을 숫자변환 전에 제외하고 lock±1purge하여 학습7344행306일. 학습특징은 MASK 허용10원시열 및 기존 파생열만. 평가 특징은 같은 온실 현재·이전 입력만, train/test 입력을 함께 정렬하되 전체평가통계를fit하지 않고 현재까지 특징만 만든다. 모델 fit은 학습행만. 잠금채점·리더보드 비중선택·외부조회·기존파일수정 금지.

정확 기준은 기존 NumPy2.5.3 재현의 prediction_details.npz baseline(원cacheSHA36fd8bef6218b158cd1d75c06f67f107430d651d2d3101832cf5fe4bb661ccd5). 기존 stateCSV는 사용하지 않는다. 원자료와 필터패키지의 학습/평가 FULL38, 학습EC, ID순서가 bit동일함은 k1_base_cache_audit_v2.json으로 확인됐다. 전달CSV는 17유효숫자·sample ID순서1440행, 잠금라벨/온도라벨을패키지에넣지않는다.

ZIP의 새폴더 추출본에서 baseline 생성기를 실제학습/예측해 CSV/ID/해시/수치일치 검산한다. 기준 대비 허용 최대절대차1e-8을 실행전고정, bit동일여부도 별도 기록. 기준CSV/ZIP/최종설명은 새 경로에만 생성, 기존 전달본 보존. ZIP 모델코드는 STATE 함수 없이 원 R3/PFN 레시피만 분리한다. clean 검산 후 최종 ZIP에검증기록을추가하고다시새폴더추출해실행코드/입력/체크포인트해시가 학습한검산본과동일함을확인한다.
