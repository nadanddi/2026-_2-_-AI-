from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,re
RUN=Path(__file__).resolve().parent
v=json.loads((RUN/'final_checks_v1.json').read_text(encoding='utf-8'));assert v['pass_all']
assert (RUN/'critic_final_recheck_v1.json').is_file()
catalog=ROOT/'공용/데이터_단서_카탈로그.md';n=max(int(x) for x in re.findall(r'\| 6\.(\d+) \|',catalog.read_text(encoding='utf-8')))+1
text='2026-10-10 집 코덱스 이미지 유사사진 확대: 개발6농장 사전동결15개TS TAR/17농장별묶음에서새408장직접decode/기존VS1000와crossJPEG SHA0. 일반색조건229는유사판정아님;자동후보60의비교판7개전량육안→약한5제외→추가55(금실38/설향17),장면41/구성요소14 정성구분과수동공개참조/이유동봉. 기존49보존+새통합104(금실83/설향21),전104현재파일SHA/decodedpixelunique104·CSV/sql집계PASS. 전추가55원본TAR독립재읽기byte/SHA/치수확인,기존49·공개21과완전pixel중복0;같은장면/농장독립/효용미확인. 공개근접추가415753 1주의표시,낮은근접hash없음이계보독립보장아님.'
phase='진행장애는Drive인터넷이아닌ledger의반복os.replace WinError5(debug실traceback). 실패v4保존→v5요청전JSONLappend flush/fsync로해소. v6부분줄개행복구준비/coverage실사용. v5첫360/15묶음781초에12분다음묶음stop,추가coverage48/2묶음별도실행→408,전체24묶음576/350GB전수완료아님. watchdogread/open/seek90초실timeout재현은미실행. 실requested검산후863615284B/payload855829573B(Drive제공자hydration량미측정). imagecache12506703937B/newtask763281564B/매추가copyguard실행. 비평가짝비교04표시이상지적→7판fresh재렌더v2/v1byte동일·직접crop caption이미지정상확인,중복조건finalchecks별도assert.'
phase=phase.replace('保존','보존')
with catalog.open('a',encoding='utf-8') as f:f.write('\n| 6.%d | **이미지 유사사진 추가55·통합104장** (2026-10-10 집 코덱스) | %s %s | 육안선정·범위제한 · `집/코덱스/analysis/image_similar_expansion_20261010_v1/final_checks_v1.json` 및 `verification_v1.json` |\n'%(n,text,phase))
for path in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-10.md']:
    with path.open('a',encoding='utf-8') as f:f.write('\n\n## 2026-10-10 집 코덱스 · 유사사진 검색 확대\n'+text+'\n'+phase+'\n카탈로그6.%d. ownlocal image_similar_expansion_20261010_v1/유사사진_통합104장(원본JPEG+선정표),공개샘플_사진별_짝비교_*_v2.jpg. 6농장metadata표시며대부분AIF00584장편향,104독립장면주장금지. 계획/중간/최종비평및재확인완료. 새증강/학습/점수/제출0·기존EC중지/다른AI파일변경0. 다음:이미지모음검토·원본그룹계보후유형별제작;전체TAR미확인분과100source/300filter확대아직미실행. 동시작업으로자동sync_end0.\n'%n)
with (RUN/'completion_record_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(catalog_id='6.%d'%n,recorded=True),f)
print('recorded 6.%d'%n)
