from pathlib import Path
p=Path(__file__).parent
s=(p/'run_v1.py').read_text(encoding='utf-8')
s=s.replace("import sys,csv,json,io,hashlib,warnings", "import sys,csv,json,io,hashlib,warnings,time")
s=s.replace("   if '--unused' in []:return", "   while not (OUT/'MIDPOINT_APPROVED.txt').exists():\n    assert datetime.now(timezone(timedelta(hours=9))).hour<15,'deadline reached during review'\n    time.sleep(2)")
(p/'run_v2.py').write_text(s,encoding='utf-8')
s=(p/'run_remote_v1.sh').read_text(encoding='utf-8').replace('run_v1.py','run_v2.py')
(p/'run_remote_v2.sh').write_text(s,encoding='utf-8',newline='\n')
