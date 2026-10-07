from pathlib import Path
import sys,csv,json,collections,hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
OUT=Path(__file__).parent;DATA=Path(env.DATA)
def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
x,y,t=map(read,['train_X.csv','train_y.csv','test_X.csv'])
yi={r['row_id']:r for r in y};xi={r['row_id']:r for r in x}
assert len(yi)==len(y) and len(xi)==len(x)
ec=[r for r in y if r['sub_ec'].strip()];assert all(r['row_id'] in xi for r in ec)
missing=[c for c in t[0] if c!='row_id' and all(r[c]=='' for r in t)]
usable=[c for c in x[0] if c!='row_id' and c not in missing]
assert len(usable)==14 and len(missing)==5
train=[xi[r['row_id']] for r in ec];byfarm=collections.Counter(r['row_id'].split('_')[0] for r in ec)
complete=sum(all(r[c]!='' for c in usable) for r in train)
missing_counts={c:sum(r[c]=='' for r in train) for c in usable}
missing_ids=set()
for c in usable:missing_ids.update(r['row_id'] for r in train if r[c]=='')
assert len(train)-len(missing_ids)==complete
assert len(ec)==sum(r['row_id'] in yi and yi[r['row_id']]['sub_ec']!='' for r in x)==sum(byfarm.values())
groups=collections.defaultdict(set)
for r in ec:
    farm,day,hour=r['row_id'].split('_');groups[(farm,day)].add(int(hour))
assert all(h==set(range(24)) for h in groups.values())
result={'scope':'competition distributed original CSV only; no external/features/OOF/predictions/synthetic additions','source_files':{n:{'path':str(DATA/n),'sha256':hashlib.sha256((DATA/n).read_bytes()).hexdigest()} for n in ['train_X.csv','train_y.csv','test_X.csv']},'training_ec_rows':len(ec),'by_farm':dict(byfarm),'farm_days':len(groups),'usable_original_input_columns':usable,'test_all_missing_excluded_columns':missing,'complete14_rows':complete,'missing_input_rows':len(missing_ids),'missing_cells_by_column':missing_counts,'test_rows':len(t),'test_missing_in_usable':{c:sum(r[c]=='' for r in t) for c in usable},'other_farms_ec_label_rows':sum(n for f,n in byfarm.items() if f not in ['F13','F47']),'join_key':'row_id','secondary_observed_label':'train_y.sub_temp exists; unavailable as observed input for evaluation rows, so excluded from vanilla EC input','raw_definition_limit':'organizer says some train inputs reconstructed/transformed without markers; original distributed data is not guaranteed entirely raw sensor observation','checks':['EC label count independent y-list / X-key lookup / farm counts agree','complete rows agree with missing-row set complement','all 400 farm-days have 24 hours'],'created_training_dataset':False,'cleaning':False,'model_training':False}
p=OUT/'vanilla_inventory_v1.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
with (ROOT/'집/코덱스/작업일지/2026-10-07.md').open('a',encoding='utf8') as f:f.write('\n## 범위 정정 EC 바닐라 데이터만 확인\n- 사용자 외부·파생·합성 제외 요청. ec_vanilla_inventory_20261007_v1/check_v1.py로 배포원본3CSV fresh 재검산: EC정답9600행(F13/F47각4800), 원입력14열, 5열MASK제외, 완전9511/결측행89, test1440행14열결측0. y-list/X-key/farm 합계·완전행/결측ID집합 독립일치. 원본은주최측복원값포함가능(표시없어분리불가). 추가학습데이터/정제/학습0. 기존카탈로그6.370과수치동일·새발견번호중복생성0.\n')
with (ROOT/'공용/HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n### 2026-10-07 집 코덱스 사용자 범위 정정 EC 바닐라 자료\n- 외부·파생·예측·합성 제외하고 배포 train_X+train_y.sub_ec row_id결합만재확인: 9600행/14입력/400기록일. test1440행. 5개100%결측열MASK제외, sub_temp는평가관측값없어바닐라EC입력제외. vanilla_inventory_v1.json에원본SHA/독립집계/결측저장. 원본복원값미표시한계유지. 학습/채택/제출0; 다음실험은이범위를기초자료로삼음.\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
