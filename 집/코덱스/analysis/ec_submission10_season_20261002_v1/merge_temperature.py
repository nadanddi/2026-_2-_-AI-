"""Teammate-only merge helper: preserve both CSV field strings, create a new file."""
from pathlib import Path
import argparse,csv,math
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--temperature',type=Path,required=True)
    p.add_argument('--ec',type=Path,default=Path(__file__).resolve().parent/'submission_10.csv')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'새 이름을 사용하세요'
    ec=rows(a.ec);temperature=rows(a.temperature)
    assert len(ec)==len(temperature)==1440
    ids=[r['row_id'] for r in ec];temp={r['row_id']:r['sub_temp'] for r in temperature}
    assert len(set(ids))==len(temp)==1440 and set(ids)==set(temp)
    assert all(math.isfinite(float(r['sub_ec'])) and math.isfinite(float(temp[r['row_id']])) for r in ec)
    with a.output.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['row_id','sub_temp','sub_ec'])
        w.writerows((r['row_id'],temp[r['row_id']],r['sub_ec']) for r in ec)
    print('1440행 새 합본 생성. EC/온도 필드 문자열 보존. 팀원 온도 모델까지 포함한 합본 재현 검산은 별도로 필요합니다.')
if __name__=='__main__':main()
