"""verify.py의 분포 사전 key 형식만 정정. 원본 보존, 새 검산 파일 출력."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
source=(HERE/'verify.py').read_text(encoding='utf-8')
assert source.count('str(float(k))')==2
source=source.replace('str(float(k))','str(k)')
source=source.replace("HERE/'verification.json'","HERE/'verification_v2.json'")
exec(compile(source,str(HERE/'verify.py'),'exec'),dict(__name__='__main__',__file__=str(__file__)))
report=json.loads((HERE/'verification_v2.json').read_text(encoding='utf-8'))
report['repair']='분포 key int를 float 문자열로 잘못 변환한 검산 코드만 str(int)로 정정; 수치/가설/원시파일 수정 없음'
report['repair_wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE/'verification_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
