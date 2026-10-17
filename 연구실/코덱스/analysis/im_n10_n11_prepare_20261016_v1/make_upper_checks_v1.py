from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'self_checks_v1.py').read_text(encoding='utf-8').replace('for n in (10,11):','for n in (10,):').replace('build_v2/n{n}','build_upper_v1/n{n}').replace('self_checks_v1.json','self_checks_upper_v1.json')
with (P/'self_checks_upper_v1.py').open('x',encoding='utf-8') as f:f.write(s)
s=(P/'validate_v4.py').read_text(encoding='utf-8')
s=s.replace("HERE/'build_v2'/f'n{n}/팜모니_나단{n}/inference.py'","HERE/('build_upper_v1' if n==10 else 'build_v2')/f'n{n}/팜모니_나단{n}/inference.py'")
s=s.replace("HERE/'build_v2/n10/","HERE/'build_upper_v1/n10/")
s=s.replace("HERE/'build_v2'/f'n{n}/expected_","HERE/('build_upper_v1' if n==10 else 'build_v2')/f'n{n}/expected_")
s=s.replace("HERE/'build_v2'/f'n{n}/actual_","HERE/('build_upper_v1' if n==10 else 'build_v2')/f'n{n}/actual_upper_")
s=s.replace('}_v1.json','}_upper_v1.json')
with (P/'validate_upper_v1.py').open('x',encoding='utf-8') as f:f.write(s)
print('UPPER_CHECK_CODE_READY')
