from pathlib import Path
out=Path(__file__).parent
source=(out/'final_checks_v1.py').read_text(encoding='utf8')
source=source.replace("'all_file_checks_v1.json'","'all_file_checks_v2.json'").replace("'final_verification_v1.json'","'final_verification_v2.json'")
exec(compile(source,str(out/'final_checks_v1.py')+' [v2 repair selection]','exec'))
