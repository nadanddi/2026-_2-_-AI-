"""사전 고정 집단의 분모/독립 분해·기존 예시 원문 점검용 요약. 새 검정 없음."""
import csv,json,math
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'row_features.csv').open(encoding='utf-8-sig',newline='')))
days=list(csv.DictReader((HERE/'day_features.csv').open(encoding='utf-8-sig',newline='')))
strict=[r for r in days if r['strict_fraction_group']=='1']
levelsse=math.fsum(float(r['level_sqerr']) for r in strict)
allmse=math.fsum(float(r['mse']) for r in days)
output=dict(strict_days=len(strict),strict_farm_counts=dict(Counter(r['farm'] for r in strict)),
    strict_section_counts=dict(Counter(r['section'] for r in strict)),
    strict_level_sse_share=levelsse/allmse,strict_level_oracle_rmse=math.sqrt((allmse-levelsse)/len(days)),
    strict_group_ec_mean=math.fsum(float(r['ec_mean']) for r in strict)/len(strict),
    strict_group_level_sqerr_mean=levelsse/len(strict),
    strict_group_level_sqerr_median=sorted(float(r['level_sqerr']) for r in strict)[len(strict)//2],
    strict_group_shape_mse_mean=math.fsum(float(r['shape_mse']) for r in strict)/len(strict),
    strict_group_days=[dict(farm=r['farm'],day=int(r['day']),ec_mean=float(r['ec_mean']),strict_linear_fraction=float(r['strict_linear_fraction']),plateau_fraction=float(r['plateau_fraction'])) for r in strict],
    strict_linear_days=[dict(farm=r['farm'],day=int(r['day']),ec_mean=float(r['ec_mean'])) for r in days if float(r['strict_linear_fraction'])>0],
    raw_decimal_digit_counts=dict(Counter(r['raw_decimal_digits'] for r in rows)),
    exact_grid01_rows=sum(int(float(r['grid01'])) for r in rows),exact_grid1_rows=sum(int(float(r['grid1'])) for r in rows),
    raw_example_rows=sum(1 for _ in csv.DictReader((HERE/'raw_examples.csv').open(encoding='utf-8-sig',newline=''))))
(HERE/'diagnostic_summary_v3.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))
