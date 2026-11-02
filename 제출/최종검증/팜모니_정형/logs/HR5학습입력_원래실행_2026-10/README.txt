HR5 학습 입력(code/pipeline_35e/EC/hr5_train_base_nosg_oof.csv)을 처음 만든 연구 실행의 기록 (팀 연구 폴더에서 복사, 2026-10-20)

순서
1. ec3_WT0_r3_member_weights_v1.py (2026-10-05, CPU)  -> ET / LightGBM / MLP 구성원 폴드 밖 예측(시드 47/1414/6464)
   로그: ec3_WT0_r3_member_weights_v1.log (.err 비어 있음)
2. ec3_WT1_dp1_removal_pfn_share_v1.py (2026-10-06)  -> TabPFN 폴드 밖 예측(문맥 5~8)
   이 스크립트는 env_extra_gpu 를 불러오고 TabPFN 을 device="cuda"(GPU)로 실행합니다(코드 사본 동봉). 폴드를 세 작업자로 나눠 돌렸습니다(wt1_worker0~2.log).
   실행한 GPU 의 모델명·드라이버 버전은 로그에 남아 있지 않습니다.
   로그: ec3_WT1_dp1_removal_pfn_share_v1*.log, wt1_worker*.log
3. ec_rv1_wt1_on_sg2_v1.py (2026-10-07, CPU)  -> nosg = clip(0.8 R3 + 0.2 PFN), DIAG10 폴드, 시드 평균 -> hr5_train_base_nosg_oof.csv 의 값
   로그: ec_rv1_wt1_on_sg2_v1.log

같은 절차를 DIAG10 폴드에 한정해 다시 실행하는 코드: code/retrain/hr5_base_oof/집/클로드/research/make_hr5_base_oof_v1.py
2026-10-17 CPU 재생성 결과: ET/LGB/MLP 는 원래 실행과 같음(최대 차 4.4e-16), TabPFN 부분만 GPU↔CPU 차이(README ⑥).
