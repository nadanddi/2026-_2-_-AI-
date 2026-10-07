"""Pre-score correction: preserve historical lock±1 and reject cross-pass spans."""
from pathlib import Path
path=Path(__file__).with_name('blk_layout_v1.py')
code=path.read_text(encoding='utf-8')
code=code.replace('eligible_days=available-locked','locked_near={(f,d+j) for f,d in locked for j in [-1,0,1]}\neligible_days=available-locked_near')
code=code.replace("if not required<=eligible_days:continue","if not required<=eligible_days:continue\n            if len({d>=179 for f,d in required})!=1:continue")
code=code.replace('training=available-hidden-gaps-locked','training=eligible_days-hidden-gaps')
code=code.replace("'hidden_both_label_ids':rowids(hidden|gaps|locked),","'hidden_both_label_ids':rowids(hidden|gaps|locked_near),\n    'locked_near_ids':rowids(available&locked_near),\n    'endpoint_anchor_ids':[{'farm':b['farm'],'query_days':b['query_days'],'left_23h':f\"{b['farm']}_{b['flank_left'][-1]:03d}_23\",'right_0h':f\"{b['farm']}_{b['flank_right'][0]:03d}_00\"} for b in selected],\n    'revision_reason':'BLK v1에서 역사 locked±1 54기록 누락 발견. 점수 계산 전 v2로 수정, v1 학습 금지',")
code=code.replace("path=HERE/'BLK_layout_v1.json'","path=HERE/'BLK_layout_v2.json'")
exec(compile(code,str(Path(__file__).resolve()),'exec'))
