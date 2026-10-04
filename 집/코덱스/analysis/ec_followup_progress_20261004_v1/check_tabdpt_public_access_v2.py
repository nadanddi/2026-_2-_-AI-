"""Anonymous official metadata/license and HEAD only; no tokens or model bytes."""
from pathlib import Path
import hashlib
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

H = Path(__file__).resolve().parent
dest = H/'tabdpt_public_access_v2.json'
assert not dest.exists()
revision = 'a5ca6e01c0fa09ec68c73e958e5199d1932abb3a'
base = 'https://huggingface.co/Layer6/TabDPT'
checks = []
def fetch(url,method='GET'):
    req = urllib.request.Request(url,method=method,headers={'User-Agent':'farmai-readonly-research/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:
        data = b'' if method=='HEAD' else r.read(2*1024*1024)
        final = urlparse(r.url)
        return dict(status=r.status,url=url,final_host=final.hostname,
                    content_length=r.headers.get('Content-Length'),
                    etag=r.headers.get('ETag'),data=data)
for name,url,method in [
    ('metadata',f'https://huggingface.co/api/models/Layer6/TabDPT/revision/{revision}?blobs=true','GET'),
    ('card',f'{base}/raw/{revision}/README.md','GET'),
    ('license',f'{base}/raw/{revision}/LICENSE','GET'),
    ('weight_head',f'{base}/resolve/{revision}/tabdpt1_3.safetensors','HEAD')]:
    try:
        rec = fetch(url,method)
        data = rec.pop('data')
        rec.update(name=name,sha256=hashlib.sha256(data).hexdigest() if data else None)
        if name=='metadata':
            meta = json.loads(data)
            rec.update(gated=meta.get('gated'),private=meta.get('private'),revision=meta.get('sha'),
                       license=meta.get('cardData',{}).get('license'))
            for item in meta.get('siblings',[]):
                if item.get('rfilename')=='tabdpt1_3.safetensors':
                    rec['weight_metadata'] = item
        elif name in ['card','license']:
            text = data.decode('utf-8')
            rec.update(apache2=('Apache License' in text and 'Version 2.0' in text) if name=='license' else ('license: apache-2.0' in text))
            (H/f'tabdpt_{name}_revision_v2.txt').write_text(text,encoding='utf-8')
        checks.append(rec)
    except Exception as error:
        checks.append(dict(name=name,url=url,status='ERROR',error=f'{type(error).__name__}: {error}'))
metadata = next(r for r in checks if r['name']=='metadata')
license = next(r for r in checks if r['name']=='license')
head = next(r for r in checks if r['name']=='weight_head')
weight = metadata.get('weight_metadata',{}).get('lfs',{})
passed = (metadata.get('status')==200 and metadata.get('gated') is False and metadata.get('private') is False
          and metadata.get('revision')==revision and license.get('apache2') is True and head.get('status')==200
          and weight.get('sha256')=='97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'
          and weight.get('size')==252233296)
res = dict(status='PASS' if passed else 'ACCESS_NOT_FULLY_VERIFIED',checks=checks,
           model_download=0,installation=0,fit=0,authentication_or_agreement_action=0)
dest.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
