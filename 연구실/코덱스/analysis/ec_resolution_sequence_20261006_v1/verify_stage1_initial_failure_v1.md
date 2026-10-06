# 독립검산 코드 초기 실패 기록

2026-10-06. verify_stage1_v1.py 첫 snapshot실행에서 reviewer코드가 train_X의F13/F47필터를누락했다. stage1_v2.py에는해당필터가올바르게구현되어있다. 다른온실까지features/vectors를처리하여 season.py vectors의len(g)==24 assert에서중단됐다. 이시점에snapshot PASS나전체PASS를선언하지않았다. stage모델학습/source/cache변경없음.

검산코드에대상온실선행필터만추가해재실행한다. 이는원모델실험결함또는replayguard불합격이아니다. 첫실행이불필요하게전체온실을처리해오래걸린원인도동일하다.
