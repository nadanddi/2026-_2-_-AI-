from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,re
RUN=Path(__file__).resolve().parent
v=json.loads((RUN/'verification_v1.json').read_text(encoding='utf-8'));assert v['pass_all']
assert (RUN/'critic_final_v1.json').is_file()
catalog=ROOT/'공용/데이터_단서_카탈로그.md'
n=max(int(x) for x in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))+1
text='2026-10-10 집 코덱스 사용자 요청 유사사진 1차 모음: VS금실2/설향3 캐시1000 공개21별 상위3 합집합56→공개참조별 육안63쌍 중8쌍/고유7장 제외→최종49(금실45/AIF005·설향4/AIF007),같은prefix7개/독립장면아님. 원본JPEG 무편집복사·전49 독립tarfile바이트/SHA/치수와CSV SQLite품종·농장·prefix 검산PASS. 원본모음/선정이유CSV/미리보기5장 및 자동후보단계비교판7개 보존. 그림체·노이즈·합성배경 효과와 작물·시설·구도 유사 분리,약한잎근접후보는구성요소유사로제한. 계획/중간/최종비평완료. guardv1 EC전체cache합산으로복사전중단→v2 image_*범위정정;이미지캐시11743407618B<16GiB. 전체350GB검색/같은원본확정/학습효용/새학습/제출0.'
with catalog.open('a',encoding='utf-8') as f:f.write('\n| 6.%d | **이미지 유사원본49장 별도 모음** (2026-10-10 집 코덱스) | %s | 육안 선택·범위 제한 · `집/코덱스/analysis/image_similar_collection_20261010_v1/final_manifest_v1.json` 및 `verification_v1.json` |\n'%(n,text))
for path in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-10.md']:
    with path.open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-10 집 코덱스 · 유사 이미지 모음\n'+text+'\n카탈로그6.%d. 산출물: 집/코덱스/local/image_similar_collection_20261010_v1/선별한_유사사진_49장. 다음: 사용자 시각검토와 더넓은자료원검색 후 증강제작. 기존EC중지/다른AI작업변경0,동시작업이므로자동sync_end0.\n'%n)
with (RUN/'completion_record_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(catalog_id='6.%d'%n,recorded=True),f)
print('recorded 6.%d'%n)
