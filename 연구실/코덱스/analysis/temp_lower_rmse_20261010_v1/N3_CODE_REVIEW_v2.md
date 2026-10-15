# N3 v2 실행 전 재검토

2026-10-10. n3_run_v2/model_v2 직접읽음. 등록bootstrapseed와실제RNG모두계획20261010으로일치하여이전차단점해소. 모델firstassert의querysub_temp거부경로에syntheticlabel987654를넣고AssertionError를catch해receipt를남기며거부되지않으면실행실패한다. 따라서이negativecall에서추가fit가발생하지않고,그뒤label삭제한frame만실predict한다.

level/shape훈련지원receipt도추가됐고예측불변성실험이아닌학습전컬럼reject검사라고comment를정정했다. alpha.005/원모델parameter/feature/키/부트스트랩계약은변경없다. **추가차단점없으며--first 진행가능.** critic새fit0/EC0/외부0. 실제firstmetadata/NPZ·rawbaselineexact/shape0차감·sourceSHA 확인은후속중간검산에서한다.
