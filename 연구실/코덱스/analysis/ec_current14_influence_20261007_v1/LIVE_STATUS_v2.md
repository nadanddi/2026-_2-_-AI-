# 현재 EC14 검증 완료·삭제 진단 실행 중

2026-10-07 연구실 코덱스. LIVE_STATUS_v1보다 최신 상태다.

baseline runner_v4 정상 종료. 신규 R3모델90개, PFN 신규0개, 전수34560예측행/360날과 3seed·실제rawensemble을 보존했다. score_v1 baseline 완료. 독립30cache/전후처리/원본SG2/후보ranking/2880일점수/96집단·fsum/PFN40문맥/실제활성SG2 섭동6건 모두 PASS, baseline_critic_v1.md 작성 완료. 1단계_검토_v1.md에 결과와 범위를 정리했다.

현재 EC14 ensemble에서 일반 환기0≥0.8 61일 RMSE0.202410, 나머지268일0.075497. 전체0.178957은 기존0.181413보다낮지만 폐쇄61일은기존0.193977보다높다. 대표실패 F47_161 실제0.646/최종1.144834·rawET1.404064, F13_112 실제0.396333/최종0.665957이남아있다. 네선택날은P1이며SG2대상아니다. 새채택/제출0.

2단계 실행: ablation_v1.py, source f570b01, 준비등록42dd5ad, 도구세션55695, 프로세스4976, 시작2026-10-07 02:21:57. 로그는 내 local/ec_current14_influence_20261007_v1/ablation_v1.log다. 현재 ablation.lock이 있고 learner 실행 중이다. source/문턱/계수 변경·재시작·부분성능채점 금지. source에는 재개기능이 없으므로 실패 시 이미생성된산출물을보존하고새버전으로검증재사용해야한다.

현재 fullforest BASE/D1/D2 3개와 sidecar는완성됐다(약251MB). 10fold×D1/D2×3seed 전체캐시/예측/receipt는아직미완료다. 전수 receipt가생긴뒤 score_v1 --ablation, audit_support_v1, 독립캐시·후처리·채점·최종혹독비평 순서로진행한다. 삭제판정은 rawET24h평균편향으로고정, 추적용BASE1fit과삭제fit수는별도기록한다. 부분완료를개선확정으로표현하지않는다.

다른AI/집worker 변경0, 원y/test_X/EL1/잠금정답 새읽기0. 사용자에게sync_end를권할때는두AI가끝난뒤1회라고안내한다.
