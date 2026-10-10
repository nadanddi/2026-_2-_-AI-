from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
RUN=Path(__file__).resolve().parent
verify=json.loads((RUN/'verification_v1.json').read_text(encoding='utf-8'));assert verify['all_checks_pass']
assert (RUN/'critic_mid_final_v1.json').is_file()
catalog=ROOT/'공용/데이터_단서_카탈로그.md';previous=catalog.read_text(encoding='utf-8')
number=max(int(n) for n in re.findall(r'^\|\s*6\.(\d+)\s*\|',previous,re.M))+1
entry=f'\n| 6.{number} | **공개샘플21 설명·원본사진 개별 카탈로그 완료** (2026-10-10 집 코덱스,사용자우선지시) | PDF3쪽렌더·21사진각원해상도 직접확인. 실제3+생성6유형각3=21,CSV/JSON/HTML/TXT ID·순서·내용/PDF쪽 일치. 원본21JPEG HTML내장byte/SHA동일,SQL독립type/label/dim집계(세로17/가로4). 유형별3사진판7개 육안확인. 공식설명/직접관찰/unknown/AI허브선정참고/증강정의 분리. 특히filter Gaussianblur/noise/SP 종류는공식명시·강도unknown,부분gen/style 원본/마스크미제공으로정확편집영역단정금지. HTML실브라우저기능은file프로토콜정책차단으로미검증/우회0. | 카탈로그내용·파일검증PASS/성능아님 | `집/코덱스/analysis/image_public_catalog_20261010_v1/공개21_이미지목록_v1.html`, `유형별_사진별_정리_v1.txt`, `verification_v1.json`, `critic_mid_final_v1.json` |\n'
with catalog.open('a',encoding='utf-8') as f:f.write(entry)
note=f'''\n\n## 2026-10-10 · 집 · 코덱스 사용자 우선 지시: 공개21 카탈로그 정리 완료
공개샘플21_설명 PDF3쪽 fresh텍스트/PNG 및 실제사진21개 각원해상도 직접확인. 공식CSV타입·라벨/PDF쪽별요약/직접관찰/미확정/AI허브선정참고/후속증강정의를21행으로정리,HTML독립이미지목록(원본JPEG내장)/CSV/JSON/TXT/유형판7개 저장. SQL독립counts7×3/라벨3:18/크기17세로4가로,HTML21원본byte/SHA/서로다른출력내용·순서일치PASS. PDF원본/사진수정0,학습/성능/제출0. 부분gen/style의실편집위치추정금지·필터공식종류와unknown강도구분. 브라우저fileURL정책차단→실HTML버튼미검증/우회0,정리파일에명시. 독립비평후기록완료. 카탈로그6.{number}. ownanalysis image_public_catalog_20261010_v1/읽는_순서_v1.txt 및검증자료.
직전유사선별진행:캐시1000전체공개21score보존/색감치수격리조건통과513(모두유사사진판정X),첫원본2둘다금실AIF005. 사용자2장제한비판반영/본제작2장제한폐기. 확대계획100source/정확filter300은선정·제작미실행/확대비평추가조건미구현. 첫imagegen실서비스2건(전체새생성+배경교체) assets_first보존,나머지8미실행. source2sameGPU/색독립검산PASS,CPU동치검산은5.1e-5차이로실패기록보존. 사용자Python필터허용상태유지. 최신요청에따라카탈로그정리를우선완료했으며다음은이자료기준실제유사후보넓은선정·장면계보/중복·조건부확대제작. 전체350GB검색/훈련완료주장X. EC기존중지유지/다른AI작업변경0. sync_end는다른AI진행상태확인후실행권장,이번자동동기화0.
'''
for p in [ROOT/'공용/HANDOFF.md',ROOT/'집/코덱스/작업일지/2026-10-10.md']:
    with p.open('a',encoding='utf-8') as f:f.write(note)
receipt={'catalog_number':f'6.{number}','verification_PASS':True,'original_source_count':21,'report_sha256':hashlib.sha256((RUN/'유형별_사진별_정리_v1.txt').read_bytes()).hexdigest(),'scope':'public21 organization priority complete; broader augmentation remains pending'}
with (RUN/'completion_record_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
