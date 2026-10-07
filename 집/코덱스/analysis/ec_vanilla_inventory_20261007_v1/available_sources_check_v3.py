import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
prior = json.loads((HERE / 'vanilla_inventory_v1.json').read_text(encoding='utf-8'))
data = {}
hashes = {}
for name, meta in prior['source_files'].items():
    path = Path(meta['path'])
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes[name] == meta['sha256'], name
    with path.open(encoding='utf-8-sig', newline='') as handle:
        data[name] = list(csv.DictReader(handle))
x, y, test = (data[n] for n in ['train_X.csv', 'train_y.csv', 'test_X.csv'])
xi = {r['row_id']: r for r in x}
yi = {r['row_id']: r for r in y}
assert len(xi) == len(x) and len(yi) == len(y)
assert not (set(yi) - set(xi))
counts = {label: sum(bool(r[label].strip()) for r in y) for label in ['sub_ec', 'sub_temp']}
for label, count in counts.items():
    joined_count = sum(bool(yi[r['row_id']][label].strip()) for r in x if r['row_id'] in yi)
    farm_counts = Counter(r['row_id'].split('_')[0] for r in y if r[label].strip())
    assert count == joined_count == sum(farm_counts.values())
missing_y = sum(r['row_id'] not in yi for r in x)
assert missing_y == len(set(xi) - set(yi)) == len(x) - len(y)
farms = set(r['row_id'].split('_')[0] for r in x)
ec_farms = set(r['row_id'].split('_')[0] for r in y if r['sub_ec'].strip())
temp_without_ec = sum(bool(r['sub_temp'].strip()) and not r['sub_ec'].strip() for r in y)
assert temp_without_ec == counts['sub_temp'] - counts['sub_ec']
result = {
    'scope': '대회 배포 원본만, 외부·예측값·파생값·합성 관측 제외',
    'counts': {'train_X': len(x), 'train_y': len(y), 'test_X': len(test),
               'farms': len(farms), 'EC_label_farms': sorted(ec_farms),
               'EC_label_rows': counts['sub_ec'], 'temperature_label_rows': counts['sub_temp'],
               'temperature_label_without_EC_rows': temp_without_ec,
               'X_rows_without_y': missing_y, 'farms_without_EC_label': len(farms - ec_farms)},
    'sources': [
        'testで観測可能な原入力14列: ' + ', '.join(prior['usable_original_input_columns']),
        'row_idの温室ID・相対記録日・時刻と原行の結合キー',
        '同じ温室の現在・過去の原入力記録（後時刻入力は禁止）',
        '公開学習記録の原入力・sub_ec・sub_tempを保持した参照候補',
        'EC正解がない49温室を含む全train_Xと公開sub_temp（補助学習の候補）',
    ],
    'constraints': [
        '上記sourcesの日本語記述は次のsources_koを正式な説明とする',
        '평가 행 실제 sub_temp는 없음. 보조 목표·공개 학습 이웃 참조를 검토할 수 있으나 성능 효과는 미검증',
        '검증 보류 행의 EC와 온도 정답은 학습·참조에서 함께 숨김. 학습 특징은 test 결측 MASK 적용',
        '공개 학습 이후 날짜 정답 참조 허용은 공용 메모리의 운영 안내 요약에 근거. 이번에 원 안내 PDF를 재확인한 것은 아님',
        '거리·평균·차분·변화율·예측 온도·복원 계절은 파생값. 원본 자료 목록과 구분',
        '상대 기록일 연속이 실제 같은 온실의 물리적 연속을 보장하지 않음',
        '제공 CSV 자체에 주최측 복원·변환 값이 포함될 수 있고 표시가 없어 순수 센서값만 분리할 수 없음',
    ],
    'sources_ko': [
        '평가에서 관측 가능한 원입력 14열', 'row_id: 온실 ID·상대 기록일·시각·결합키',
        '동일 온실 현재·과거 원입력 기록', '공개 학습 이웃 원입력·EC·온도 정답',
        'EC 무정답 온실을 포함한 전체 train_X와 공개 온도 정답',
    ],
    'test_all_missing_columns': prior['test_all_missing_excluded_columns'],
    'sha256': hashes,
    'verification': '정답별 y 직접 집계/X ID 결합/온실 합계 일치; y 미대응 행수 세 방법 일치; 원본 SHA 일치',
    'performance_tested': False,
}
out = HERE / 'available_sources_v3.json'
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
note = '\n## EC 바닐라 원자료 추가 누락 점검\n- available_sources_check_v3.py: 원본 SHA 재확인, EC/온도 정답 수를 직접 집계·ID 결합·온실 합계로 교차검산. 공개 온도 304965행, 이 중 EC 없는 온도 정답 295365행, X만 있고 y 없는 5025행. 49온실을 단순 무정답으로 부르지 않도록 정정. 동일온실 입력 이력, 유사 학습 이웃, row_id 포함. 성능 효과·물리적 일차 연속성은 미확정. 근거 available_sources_v3.json.\n'
with (ROOT / '집/코덱스/작업일지/2026-10-07.md').open('a', encoding='utf-8') as f:
    f.write(note)
with (ROOT / '공용/HANDOFF.md').open('a', encoding='utf-8') as f:
    f.write('\n### 2026-10-07 집 코덱스 바닐라 자료 누락 재점검\n- 공개 sub_temp 304965행 중 EC 없는 온도 정답295365행도 EC 보조학습/공개 참조 후보. 전체 train_X309990행·51온실, y미대응5025행. 49온실은 EC무정답이지 온도까지 무정답은 아님. 원본 SHA/독립 세 집계 PASS, available_sources_v3.json. 평가 실제온도 부재·양정답 검증보류·MASK·입력시각 제한 준수. 새학습/성능판정/제출0.\n')
print(json.dumps(result['counts'], ensure_ascii=False))
