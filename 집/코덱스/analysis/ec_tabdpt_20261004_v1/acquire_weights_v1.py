"""Anonymous fixed public safetensors acquisition; no agreement, auth, or uploads."""
from pathlib import Path
import hashlib
import json
import urllib.request

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
ACCESS = ROOT/'집/코덱스/analysis/ec_followup_progress_20261004_v1/tabdpt_public_access_v2.json'
assert json.loads(ACCESS.read_text(encoding='utf-8'))['status']=='PASS'
OUT = ROOT/'집/코덱스/local/tabdpt130_cpu_v1/weights'
OUT.mkdir(parents=True,exist_ok=True)
target = OUT/'tabdpt1_3.safetensors'
partial = OUT/'tabdpt1_3.safetensors.part'
receipt = H/'weight_receipt_v1.json'
assert not target.exists() and not partial.exists() and not receipt.exists()
expected_size = 252233296
expected_sha = '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'
url = 'https://huggingface.co/Layer6/TabDPT/resolve/a5ca6e01c0fa09ec68c73e958e5199d1932abb3a/tabdpt1_3.safetensors'
request = urllib.request.Request(url,headers={'User-Agent':'farmai-public-research/1.0'})
sha = hashlib.sha256()
size = 0
with urllib.request.urlopen(request,timeout=60) as stream, partial.open('xb') as output:
    assert stream.status==200
    assert int(stream.headers['Content-Length'])==expected_size
    while True:
        block = stream.read(8*1024*1024)
        if not block:
            break
        output.write(block)
        sha.update(block)
        size += len(block)
        assert size <= expected_size
assert size==expected_size and sha.hexdigest()==expected_sha
partial.rename(target)
result = dict(status='PASS',repository='Layer6/TabDPT',revision='a5ca6e01c0fa09ec68c73e958e5199d1932abb3a',
              filename=target.name,bytes=size,sha256=sha.hexdigest(),anonymous=True,
              authentication_or_agreement_action=0,data_upload=0,fit=0)
receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
