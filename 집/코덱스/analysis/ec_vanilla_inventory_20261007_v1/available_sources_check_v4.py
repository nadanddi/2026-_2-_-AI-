import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
prior = json.loads((HERE / 'vanilla_inventory_v1.json').read_text(encoding='utf-8'))
data, hashes = {}, {}
for name, meta in prior['source_files'].items():
    path = Path(meta['path'])
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes[name] == meta['sha256']
    with path.open(encoding='utf-8-sig', newline='') as handle:
        data[name] = list(csv.DictReader(handle))
x, y, test = (data[n] for n in ['train_X.csv', 'train_y.csv', 'test_X.csv'])
xi, yi = ({r['row_id']: r for r in rows} for rows in [x, y])
assert len(xi) == len(x) and len(yi) == len(y)
assert set(yi) <= set(xi)
counts = {}
for label in ['sub_ec', 'sub_temp']:
    direct = sum(bool(r[label].strip()) for r in y)
    joined = sum(bool(yi[key][label].strip()) for key in xi if key in yi)
    farms = Counter(r['row_id'].split('_')[0] for r in y if r[label].strip())
    assert direct == joined == sum(farms.values())
    counts[label] = direct
missing_y = len(set(xi) - set(yi))
assert missing_y == sum(r['row_id'] not in yi for r in x) == len(x) - len(y)
temp_only = sum(bool(r['sub_temp'].strip()) and not r['sub_ec'].strip() for r in y)
assert all(r['sub_temp'].strip() for r in y if r['sub_ec'].strip())
assert temp_only == counts['sub_temp'] - counts['sub_ec']
result = {
    'scope': '대회 배포 원본만. 외부·예측·파생·합성 자료 제외',
    'counts': {'train_X': len(x), 'train_y': len(y), 'test_X': len(test),
               'EC_label_rows': counts['sub_ec'], 'temperature_label_rows': counts['sub_temp'],
               'temperature_label_without_EC_rows': temp_only, 'X_rows_without_y': missing_y},
    'sources': ['평가에서 관측 가능한 원입력 14열', 'row_id: 온실·상대 기록일·시각·결합키',
                '동일 온실 현재·과거 원입력', '공개 학습 이웃의 원입력과 EC·온도 정답',
                'EC 무정답 온실을 포함한 전체 train_X와 공개 온도 정답'],
    'raw_columns': prior['usable_original_input_columns'],
    'masked_columns': prior['test_all_missing_excluded_columns'],
    'constraints': ['평가 행 실제 온도는 미제공. 공개 온도는 보조 목표·학습 이웃 참조 후보',
                    '검증 보류 행의 EC·온도 정답은 함께 숨김',
                    '평가 입력은 동일 온실 현재·과거만. test 전체 통계 fit 금지',
                    '공개 학습 이후 날짜 정답 참조 허용은 운영 안내 저장 요약 근거',
                    '유사도·거리·평균·차분·보정·온도 예측은 파생값',
                    '상대 기록일 연속은 물리적 연속을 보장하지 않음',
                    '배포 원본의 주최측 복원값은 표시가 없어 순수 센서값과 분리 불가'],
    'sha256': hashes,
    'verification': '정답 수 직접집계/ID결합/온실합계 및 y없는 행 세 방법 일치',
    'performance_tested': False,
    'record_correction': 'v3 결과 존재를 기록했으나 이번 확인에서 파일 부재. v4를 실제 실행함',
}
out = HERE / 'available_sources_v4.json'
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
note = '\n### 2026-10-07 집 코덱스 바닐라 누락·기록 불일치 정정\n- v3 결과 저장 주장과 달리 실제 파일 부재 확인. available_sources_check_v4.py를 실제 실행해 원본 SHA와 정답/ID/온실 합계를 교차검산. 공개 온도·EC 무정답 온실 원입력·동일온실 이력·학습 이웃·row_id를 포함한 available_sources_v4.json 저장. 성능 효과 미검증, 외부·파생 제외 유지.\n'
for path in [ROOT / '집/코덱스/작업일지/2026-10-07.md', ROOT / '공용/HANDOFF.md']:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(note)
print(json.dumps(result['counts'], ensure_ascii=False))
