import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,hashlib,collections
from PIL import Image,ImageChops,JpegImagePlugin
RUN=Path(__file__).resolve().parent
o=json.loads((RUN/'qa_manifest_v1.json').read_text(encoding='utf-8'));sources={r['image_id']:r for r in json.loads((RUN/'prepared_sources_v1.json').read_text(encoding='utf-8'))}
checks=[];final_hashes=[]
def count(im):return sum(any(v) for v in im.getdata())
for r in o['rows']:
    s=sources[r['source_id']]
    for path,digest in [(r['path'],r['sha256']),(r['pre_path'],r['pre_sha256']),(s['qa_source_path'],s['qa_source_sha256'])]:assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    with Image.open(r['pre_path']) as im:pre=im.convert('RGB')
    with Image.open(s['qa_source_path']) as im:source=im.convert('RGB')
    real=next(x for x in o['rows'] if x['source_id']==r['source_id'] and x['label']==0)
    with Image.open(real['path']) as im:reference=im.convert('RGB');qt=im.quantization
    with Image.open(r['path']) as im:final=im.convert('RGB');assert im.size==(512,512) and im.quantization==qt and JpegImagePlugin.get_sampling(im)==2
    pre_diff=ImageChops.difference(pre,source);final_diff=ImageChops.difference(final,reference)
    assert count(pre_diff)==r['changed_pixels_pre'] and count(final_diff)==r['changed_pixels_final']
    x0,y0,x1,y1=s['content_bbox'];pad=Image.new('L',(512,512),255);pad.paste(0,(x0,y0,x1,y1));assert count(ImageChops.multiply(pre_diff,pad.convert('RGB')))==0
    farpad=Image.new('L',(512,512),255);farpad.paste(0,(max(0,x0-16),max(0,y0-16),min(512,x1+16),min(512,y1+16)));assert count(ImageChops.multiply(final_diff,farpad.convert('RGB')))==r['far_pad_changed_final']==0
    rgb=list(final.crop((x0,y0,x1,y1)).getdata());magenta=sum(red-green>20 and blue-green>20 for red,green,blue in rgb)/len(rgb);assert abs(magenta-r['magenta_fraction_on_content'])<1e-12
    if r['mask_path']:
        with Image.open(r['mask_path']) as im:mask=im.convert('L')
        assert count(ImageChops.multiply(final_diff,ImageChops.invert(mask).convert('RGB')))==r['outside_mask_changed_final_JPEG']
        outside=ImageChops.multiply(pre_diff,ImageChops.invert(mask).convert('RGB'));assert count(outside)==0
        halo=Image.new('L',(512,512));halo.paste(255,(144,144,368,368));assert count(ImageChops.multiply(final_diff,ImageChops.invert(halo).convert('RGB')))==0
    assert r['label']==int(r['positive_type']!='real') and (r['label']==0 or count(final_diff)>0)
    final_hashes.append(r['sha256']);checks.append({'sample_id':r['sample_id'],'sha_shape_quantization_change_mask_checks':True})
assert len(set(final_hashes))==14
result={'rows':len(checks),'checks':checks,'all_checks_pass':True,'type_counts':dict(collections.Counter(r['positive_type'] for r in o['rows'])),'label_counts':dict(collections.Counter(r['label'] for r in o['rows'])),'independent_method':'PIL ImageChops and Python tuple pixel counting versus NumPy assembly; fresh file SHA','added_checks':['pre pad equality','JPEG far pad equality','JPEG mask spill count','magenta pixel fraction','JPEG subsampling=2'],'source_count':len(sources),'scope':'production/file QA only; real no-edit assumption and type representativeness not authenticated'}
with (RUN/'qa_verification_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
