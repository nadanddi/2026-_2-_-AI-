from pathlib import Path
import ast
here=Path(__file__).resolve().parent
source=(here/'register_ch2_BLK_v1.py').read_text(encoding='utf-8')
source=source.replace('register_ch2_BLK_v1.py','register_ch2_BLK_v2.py').replace('run_ch2_BLK_v1','run_ch2_BLK_v2')
source=source.replace('CH2_BLK_registration_v1.json','CH2_BLK_registration_v2.json').replace('feature_candidates_v6.csv','feature_candidates_v7.csv')
source=source.replace("difference='원 slope", "difference_from_previous='원 slope").replace("result='UNTESTED'","performance='UNTESTED'")
source=source.replace("if 'formula' in fields:row['formula']", "if 'parameters' in fields:row['parameters']")
ast.parse(source);out=here/'register_ch2_BLK_v2.py';assert not out.exists();out.write_text(source,encoding='utf-8')
runner=(here/'run_ch2_BLK_v1.py').read_text(encoding='utf-8').replace('CH2_BLK_registration_v1.json','CH2_BLK_registration_v2.json')
ast.parse(runner);out=here/'run_ch2_BLK_v2.py';assert not out.exists();out.write_text(runner,encoding='utf-8')
print('Created registrar2/runner2, preserving failed registrar1/partial CSV6')
