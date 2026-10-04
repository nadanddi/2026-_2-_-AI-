from pathlib import Path
H=Path(__file__).resolve().parent
src=(H/'diagnose_cases_v1.py').read_text(encoding='utf-8')
src=src.replace('case_diagnosis_v1.json','case_diagnosis_v2.json')
src=src.replace('records.append(dict(farm=','records.append(dict(negative_changed_hours=sum(z<0 for z in delta),positive_changed_hours=sum(z>0 for z in delta),farm=')
src=src.replace('out[mode]=dict(selected_cases=',"out[mode]=dict(direction_by_segment={seg:{'changed_hours':sum(r['changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high')),'negative_hours':sum(r['negative_changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high')),'positive_hours':sum(r['positive_changed_hours'] for r in records if (r['ymean']>=1)==(seg=='high'))} for seg in ['high','ordinary']},selected_cases=")
with (H/'diagnose_cases_v2.py').open('x',encoding='utf-8') as f:f.write(src)
