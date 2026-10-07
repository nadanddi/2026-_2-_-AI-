"""Fix report rendering in new version and connect actual post-score criticism."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
text=(HERE/'DOMAIN24_BLK_진단보고서_v1.md').read_text(encoding='utf-8')
lines=text.splitlines()
segment_line=next(line for line in lines if line.startswith('실제 저장 세그먼트:'))
lines.remove(segment_line)
header=next(i for i,line in enumerate(lines) if line.startswith('| 범위 | 후보 | 일반'))
# Keep the header, separator and all48 body rows contiguous.
while lines[header+2]=='':lines.pop(header+2)
lines[header:header]=[segment_line,'']
critique=HERE/'critique_DOMAIN24_scored_diagnostics_v1.md'
assert critique.exists()
text='\n'.join(lines)+'\n'
text+='''
## 实际 사후 독립 비평

critique_DOMAIN24_scored_diagnostics_v1.md가 실제 채점 후 판정을 점검했다. QUERY_ROLE은 평균 감소22/24·전 시드 감소5/24, RAW_PASS는21/24·5/24이지만 두 범위 모두통과0이다. D22 QUERY는 P(worse)=0.0018499075로 보정기준0.025/54를 넘는다. 일부 개별95% CI가 음수여도 이를 보정된 통과로 해석하지 않는다.

D23 QUERY는 일반 RMSE차 +0.0000842·고EC -0.0083847이고 효과가 큰 한 블록에 의존한다. 다만 도메인24 전체를 CH2처럼 단일 블록에서만 바뀐 결과로 일반화하지 않는다. 이 수치는 이미 노출된 표본의 사후 진단이며 후보 정책·문턱을 조정하는 근거로 쓰지 않는다.
'''.replace('实际','실제')
with (HERE/'DOMAIN24_BLK_진단보고서_v2.md').open('x',encoding='utf-8') as h:h.write(text)
for part in ['| 순위 |','| 범위 |']:
    ls=text.splitlines();i=next(i for i,l in enumerate(ls) if l.startswith(part))
    assert ls[i+1].startswith('|---') and ls[i+2].startswith('| ')
record={'status':'DOMAIN_DIAGNOSTIC_POSTSCORE_REVIEW_AND_REPORT_RENDER_FIX',
    'source_report_sha256':hashlib.sha256((HERE/'DOMAIN24_BLK_진단보고서_v1.md').read_bytes()).hexdigest(),
    'report_v2_sha256':hashlib.sha256((HERE/'DOMAIN24_BLK_진단보고서_v2.md').read_bytes()).hexdigest(),
    'critique_sha256':hashlib.sha256(critique.read_bytes()).hexdigest(),
    'score_or_threshold_changes':False,'whole_goal_complete':False}
with (HERE/'DOMAIN24_postscore_report_receipt_v2.json').open('x',encoding='utf-8') as h:json.dump(record,h,ensure_ascii=False,indent=2)
note='''
### v18 후속 독립 사후 비평 완료
- critique_DOMAIN24_scored_diagnostics_v1.md 실제작성:QUERY 평균감소22/24·전seed감소5/24/통과0,RAW21/5/0. D22 QUERY p.0018499도54보정FAIL. 도메인전체효과를CH2단일block으로일반화금지. 보고서table형식새v2로수정·v1보존/점수정책변경0.
- critique_DOMAIN24_original_raw_plan_v2.md:OR01/OR02 source닫힘,OR03실행등록·runtime·통계코드/draw봉인 여전히남음. DOMAIN24_original_statistics_plan_v1.json은prospective초안/fit등록아님. 원66prep live77114·다음원handle확인부터. 전범위목표계속미완료.
'''
progress=HERE/'PROGRESS.md'
progress.write_text(note+'\n'+progress.read_text(encoding='utf-8'),encoding='utf-8')
for path in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-07.md']:
    with path.open('a',encoding='utf-8') as h:h.write(note)
print('Report2 table adjacency verified; postscore review linked, scoring unchanged')
