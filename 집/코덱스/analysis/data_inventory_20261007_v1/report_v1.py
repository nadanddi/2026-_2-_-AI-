from pathlib import Path
import json,collections
OUT=Path(__file__).parent
def load(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def write(n,text):
    p=OUT/n
    if p.exists():raise FileExistsError(p)
    p.write_text(text,encoding='utf8')
v=load('final_verification_v1.json');inv=load('inventory_summary_v1.json');cp=load('competition_profile_v1.json');cv=load('competition_join_and_independent_v1.json');checks=load('all_file_checks_v1.json')
lines=['# 현재 활용 가능한 데이터 전수 검사','',
'2026년 10월 7일 집 코덱스. 동기화 직후 저장소에 존재한 데이터와 산출물 목록을 기준으로 검사했다. 대회 EC에 직접 대응하는 정답은 F13·F47의 9,600행이다. 공개 외부자료는 추가 관계 조사에 사용할 후보가 있으나, 배지 EC 정답과 센서 정의·척도가 같은지 확인되지 않아 직접 합치는 판단은 보류한다. 기존 예측·특징·검증 캐시는 새로운 관측자료가 아니다.','',
'## 검사 범위','',
f"저장소의 데이터·결과·문서 {inv['files']:,}개, {inv['bytes']:,}바이트를 목록에 고정했다. CSV {inv['formats']['.csv']['files']:,}개는 모든 행을 읽어 열·결측·값 범위·완전 중복·ID 중복을 검사했다. JSON은 구문을 검사했고, 숫자 배열은 전체 내용을 읽어 NaN·무한값을 확인했다. ZIP은 전체 항목의 CRC를 검사했다. 같은 원본의 사본, 압축 백업, 캐시를 독립 관측표본으로 더하지 않는다.",'',
'Python 기본 환경은 pandas를 사용할 수 없어서 첫 시도 실패 기록을 보존하고 앱 제공 Python으로 다시 검사했다. 저장소 bootstrap인 env를 사용했으며 원본 삭제·결측 대체·단위 변경·학습·성능 채점은 모두 0회다.','',
'| 분류 | 파일 수 | 해석 |','|---|---:|---|']
meaning={'competition_raw':'대회 정형 배포 CSV','competition_other':'공고·설명서·이미지 미션 자료','public_external_raw':'공개 외부 원본 ZIP·XLSX·HWP','derived_cache':'기존 특징·OOF·예측·모델 캐시','derived_or_document':'실험 결과·메타데이터·기타 문서','submission_artifact':'기존 제출·재현 산출물','backup':'작업 백업'}
for key,n in inv['categories'].items():lines.append(f'| {key} | {n:,} | {meaning[key]} |')
lines+=['',
'인터넷 전체나 PC 전체를 전수 조사한 결과는 아니다. 개발 환경·Git·junction 중복 경로·이번 검사 폴더는 목록에서 제외했다. 다운로드가 확인되지 않은 시설원예·혁신밸리 시간 원본은 가용 원본으로 세지 않았다. 과거 비공개 대회 자료는 사용자 금지 규칙에 따라 열거나 복사하지 않았다. 모델 직렬화 파일·객체형 배열은 내용 역직렬화 대상에서 제외했고 HWP는 파일·해시만 검사했다. PDF는 목록과 대회 문제설명서 본문을 확인했다.','',
'## 대회 원본','',
'| 파일 | 행 수 | 검사 결과 |','|---|---:|---|',
f"| train_X.csv | {cp['train_X.csv']['rows']:,} | 51온실, 입력 19열·row_id, 완전 중복·ID 중복 0 |",
f"| train_y.csv | {cp['train_y.csv']['rows']:,} | 배지 온도 {cv['manual']['train_y.csv']['sub_temp']['n']:,}개, 배지 EC {cv['manual']['train_y.csv']['sub_ec']['n']:,}개 |",
'| test_X.csv | 1,440 | F13·F47 각 30기록일, 사용 가능 14열의 결측 0 |',
'| sample_submission.csv | 1,440 | 평가 ID·순서 일치; 정답이 아니라 예시값 |','',
'입력에만 있는 ID는 5,025개이고 정답에만 있는 ID는 0개다. 행 수와 순서가 서로 달라 반드시 row_id로 결합해야 한다. 독립 csv.DictReader와 pandas가 행 수·고유 ID·열별 결측·정답 범위에서 일치했고, 온라인대회자료 ZIP 속 네 CSV와 풀린 원본의 SHA256도 모두 같다. 신뢰도는 높음이며, 정답 없는 입력에 정답을 새로 만들어 넣은 것은 아니다.','',
'### EC에 대응하는 기록','',
'F13·F47에 입력과 EC 정답이 각각 4,800행씩 있다. 합계 9,600행·400기록일이고 모든 기록일에 00~23시가 있다. 일반적인 독립 표본 9,600개로 간주하지 않는다. 기록일과 온실·동의 관계는 기존 카탈로그 1.15·1.16의 한계를 유지한다. 상대 일차는 실제 달력 날짜가 아니다.','',
f"배지 EC 범위는 {cv['manual']['train_y.csv']['sub_ec']['min']}~{cv['manual']['train_y.csv']['sub_ec']['max']} dS/m이고 0·음수는 각각 {cv['manual']['train_y.csv']['sub_ec']['zero']}·{cv['manual']['train_y.csv']['sub_ec']['negative']}행이다. 원자료의 정상 범위가 외부자료의 센서 척도와 같다는 증거는 아니다.",'',
'실내 온도 결측 73셀, 습도 67셀, CO₂ 77셀이다. 14열 모두 채워진 행은 9,511/9,600행이어서 하나 이상 결측인 행은 89행이다. 결측행을 삭제하거나 채우지 않았다. 상대습도·구동기에서 0~100 범위 밖의 값은 0개, 이 14열의 비유한 숫자는 0개다. 음수 외기 온도는 정상적인 관측 가능성이 있어 오류로 분류하지 않는다.','',
'평가에서 완전히 빈 열은 in_rad, act_side, act_valve, act_cool, act_pump의 5개다. 학습 특징에도 같은 MASK를 적용해야 한다. F13·F47 외 49온실의 공개 온도 자료는 존재하지만 EC 정답은 없다. 이를 EC 지도학습 정답으로 취급할 수 없다.','',
'| 온실 | 학습 기록일 | 평가 기록일 | 전체 상대 일차 범위 | 범위 안 미제공 일차 |','|---|---:|---:|---|---|']
for r in cv['train_test_day_coverage']:lines.append(f"| {r['farm']} | {r['train_days']} | {r['test_days']} | {r['range'][0]}~{r['range'][1]} | {', '.join(map(str,r['missing_day_numbers_in_range']))} |")
lines+=['','학습·평가 간 ID와 기록일 중복은 0개다. 미제공 일차를 보간해 실제 관측이나 실제 날짜로 취급하지 않는다.','',
'## 공개 외부 원본','',
f"기업 자료 ZIP 9개, AI2024 ZIP 1개, 농진청2022 XLSX 1개, 설명 HWP 1개를 확인했다. 읽은 표는 {len(load('external_independent_v1.json'))+len(load('xlsx_independent_v1.json'))}개이고 전체 작물을 합한 원표 행 수는 {v['external_table_rows']:,}행이다. 긴 형식의 항목별 센서행을 독립 시간표본이나 EC 정답 개수로 혼동하지 않는다. 12원본의 현재 SHA256은 10월 4일 원본 보존 기록과 일치한다.",'',
'| 자료 | 딸기 농가·센서행 | EC 품질과 직접 활용 조건 |','|---|---|---|']
names={'2022_11_ds.zip':'팜커넥트 2022','2022_1_ds.zip':'골든플래닛 2022','2022_9_ds.zip':'이레아이에스 2022','2023_13_ds.zip':'그린씨에스 2023','2023_17_ds.zip':'팜한농 2023','2024_19_ds.zip':'금화이엔에스 2024','2024_20_ds.zip':'더아이엠씨 2024','2024_21_ds.zip':'정원에스에프에이 2024','2024_23_ds.zip':'팜한농 2024'}
uses={'2022_11_ds.zip':'토양센서 정의 확인 뒤 관계 조사; 고EC 크기 직접 학습 범위는 부족','2022_1_ds.zip':'표시 단위·척도 확인 전 EC 직접 병합 보류','2022_9_ds.zip':'배액 EC를 배지 EC로 치환하지 않음; 관수·배지무게 관계 조사 후보','2023_13_ds.zip':'천창·커튼·냉방 관계 조사 후보; 중복과 공급EC/pH 정의 확인','2023_17_ds.zip':'0·정수값·극단값 및 농가별 척도 확인 전 EC 병합 보류','2024_19_ds.zip':'음수 EC·극단값과 함수율 오류 확인 전 EC 병합 보류','2024_20_ds.zip':'측창과 환경 관계 조사 후보; 배지EC 정답 없음','2024_21_ds.zip':'EC가 전부0인 원본으로 EC 지도학습 보류','2024_23_ds.zip':'배액EC 전부0; 토양EC 해상도·척도 확인 전 병합 보류'}
external=[]
for filename,name in names.items():
    stem=Path(filename).stem.replace(' ','_');d=load('external_'+stem+'_v1.json');m=d['members'][0];s=m['sensor'];ecs=[g for g in s['groups'] if 'ec' in g['item'].lower()]
    text=[]
    for g in ecs:text.append(f"{g['item']} {g['n']:,}행, 0 {g['zero']:,}({g['zero']/g['n']:.2%}), 음수 {g['negative']:,}, 범위 {g.get('min')}~{g.get('max')}")
    detail='; '.join(text) if text else 'EC 항목 없음'
    lines.append(f"| {name} ({filename}) | {s['farms']}농가·{s['strawberry_rows']:,}행 | {detail}. {uses[filename]} |")
    external.append({'file':filename,'name':name,'rows':m['rows'],'strawberry_rows':s['strawberry_rows'],'farms':s['farms'],'ec':ecs,'use':uses[filename]})
lines+=['','AI2024는 환경 110,628행·생육 4,200행이다. 환경의 급액 EC 12,292행은 0.2~4.3이고 배지 EC는 없다. 농진청2022는 전체 833,135행 중 딸기 159,187행·42농가이며 토양 온도는 23,491행 채워져 있다. EC·함수율·구동기 열은 없다. 토양 온도를 배지 온도로 자동 치환하지 않는다.','',
'그린씨에스2023 원표는 완전 중복 108,621행, 농진청2022 원표는 4,895행이다. 농가·항목·시간 중복도 기록했지만, 여러 센서 채널을 합쳐 센 이상 여부는 세부 센서 ID와 함께 판단해야 한다. 시간 간격과 1시간 초과 공백은 각 농가·항목별 JSON에 보존했다. 비정상 간격이라고 자동 삭제하지 않았다.','',
'외부 CSV의 모든 행을 다시 읽어 전체·딸기 행 수, 농가 수, 항목별 EC 0·음수 개수를 pandas와 csv.DictReader로 대조했다. XLSX는 openpyxl 순회로 전체·딸기 행 수와 농가 수·토양온도 채움 개수를 다시 계산했다. 수치의 신뢰도는 높음이나, 농가명·센서 정의·공개 이용조건이 대회 데이터와 호환되는지는 별도 조건이다. 외부자료가 대회와 같은 농장이라는 판단은 하지 않았다.','',
'## 기존 캐시와 파일 오류','',
f"전체 파일 검사 상태: {dict(v['all_file_statuses'])}. 숫자 배열·압축 검사는 {sum(v['array_archive_statuses'].values()):,}개다. ZIP {inv['formats']['.zip']['files']}개는 CRC 검사에서 손상 항목이 없다. 객체형 배열 {v['object_fields_not_deserialized']}개와 객체형 NPY 1개는 내용 역직렬화를 하지 않았다. pickle·joblib 모델은 가용 파일 목록만 확인했다.",'',
f"NaN 또는 무한값이 있는 숫자 배열은 {v['array_fields_with_nan_or_inf']}개, 해당 파일은 {v['array_files_with_nan_or_inf']}개다. 무한값 배열은 {v['array_fields_with_inf']}개다. OOF 배열의 검증 밖 자리, inner_gap, 입력 결측 등이 포함되므로 NaN 자체로 캐시 손상이라고 결론 내리지 않는다. 모델 재사용 전에는 검증행 ID·유효 마스크·문맥·학습 범위·원본 SHA·완료 receipt를 해당 실험별로 대조해야 한다.",'',
'확장자가 NPZ인 파일 11개는 실제 NPZ로 읽히지 않는다. 그중 10개는 이름과 내용에서 합성 오류검사/미완성 예시로 식별됐다. 실제 캐시 1개는 아래와 같다.','',
'- `집/코덱스/analysis/local/ec_baseline_cache/6fb2fb99f59aa1abc8b707848f25e938c8d67ec42c516667993c5a7307c6af07.npz`: 62,418바이트가 모두 0. 현재 파일로는 재사용 불가이며, 필요할 때 해당 실험 규칙으로 새 캐시를 생성해야 한다. 원본 보존·삭제 0.','',
'이 오류가 현재 채택 모델의 결과를 무효화한다는 증거는 없다. 해당 캐시를 참조했는지 확인하지 않은 채 모든 기존 성능을 부정하지 않는다. 이번 검사는 모델별 성능 재검증이나 전체 코드의 인과성 인증이 아니다.','',
'## 활용 판단과 남은 조건','',
'대회 공개 EC 정답과 같은 온실의 허용 입력이 직접 학습의 기준 자료다. 다른 온실의 무정답 입력은 EC 라벨이 아니며, 기존 예측값도 실제 관측 EC가 아니다. 공개 외부자료는 관계 조사·표현 학습 후보로 남지만, 센서 정의·단위·품질·출처 및 이용조건을 정리한 뒤 대회 학습과 분리해 효과를 검증해야 한다. 이번 검사는 외부자료로 성능이 오르거나 추가 배지 EC 정답을 확보했다는 증거를 만들지 않았다.','',
'문제설명서 8.1은 권한 있는 공개자료의 학습 활용과 출처·이용조건·활용방법 기록을 요구하고, 평가 중 원격조회·다운로드·외부 추론과 test_X 외부전송을 금지한다. 시간별 특징은 같은 온실의 현재·이전 입력만 사용한다. 전체 공개 train_y 참조 범위는 저장소의 10월 5일 운영 안내 요약을 유지하되, 이번에는 그 안내 원문을 새로 확인하지 않았다.','',
'저장소는 검사 중에도 다른 실험 산출물이 생성되는 공동 작업 환경이다. 목록은 동기화 직후 스냅샷이며, 이후 새 파일과 변경은 별도 차이 목록으로 남긴다. 원본 해시는 마지막에 재검산했다.','',
'## 반론 검토','',
'| 주장 | 반론과 판단 | 신뢰도 |','|---|---|---|',
'| EC 정답 9,600개 | 온도 정답이 많은 다른 농가를 EC에 더할 수 있는가? sub_ec 비결측을 원본으로 독립 집계해 9,600행만 확인 | 높음 |',
'| 외부 EC 직접 병합 보류 | 이름과 dS/m 단위만 같으면 되는가? 토양·배액·급액은 측정 대상이 다르고 일부 척도·품질 문제가 있어 대응 확인 필요 | 높음(조건), 효용 미검증 |',
'| 실제 캐시 1개 재사용 불가 | object pickle라 읽기만 막힌 것인가? 해당 NPZ는 전체 바이트가0임을 직접 확인 | 높음 |',
'| 배열 NaN은 모두 손상이 아님 | 유효 검증 자리에서도 NaN일 수 있지 않은가? 실험별 ID·마스크 대조 전 사용 가능 인증을 하지 않음 | 높음(한계) |',
'| 파일 12,101개 검사 | 모두 독립 원본인가, 인터넷 전체인가? 사본·캐시 포함 저장소 스냅샷이며 원본과 별도 분류 | 높음(범위) |','',
'## 실행 근거','',
'- `audit_v1.py`: 목록 고정·대회 원본·외부 표·전체 CSV/JSON/배열 헤더 검사. 앱 제공 Python에서 primary, external, derived 모드 실행.',
'- `verify_v1.py`: 대회 ID 결합·51농가별 가용 정보·독립 CSV 수치 및 외부 CSV 재검산.',
'- `xlsx_independent_v1.py`: XLSX 별도 순회 재검산.',
'- `array_archive_verify_v1.py`: 숫자 배열 전체값 및 ZIP 전체 항목 CRC.',
'- `final_checks_v1.py`, `final_verification_v1.json`: 원본 SHA 재계산·배포 ZIP과 원본 대조·전체 결과 대조·오류 분류.',
'- `inventory_v1.json`, `all_file_checks_v1.json`: 파일별 전체 목록과 표 검사. `external_*_v1.json`: 원본별 모든 열·항목·농가·시간 프로필.',
'- `competition_join_and_independent_v1.json`: 51농가별 라벨·입력 가용성·기간 및 이상값 검사.',
'- `sync_rebase_metadata_backup/autostash`: 동기화를 막던 오래된 Git 메타데이터 보존. 같은 커밋은 Git stash에도 보존. 이후 sync_start 3단계 정상 완료.','']
write('전수검사_보고서_v1.md','\n'.join(lines))
write('활용가능_외부자료_v1.json',json.dumps(external,ensure_ascii=False,indent=2))
print('REPORT_WRITTEN',flush=True)
