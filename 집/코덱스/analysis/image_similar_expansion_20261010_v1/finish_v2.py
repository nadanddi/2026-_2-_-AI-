from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,hashlib,shutil,collections
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
preview=json.loads((RUN/'scan_expanded_v2_preview_v1.json').read_text(encoding='utf-8'))
review=json.loads((RUN/'visual_review_v1.json').read_text(encoding='utf-8'))
assert {r['sha256'] for r in preview['rows']}==set(review['decisions'])
chosen=[r for r in preview['rows'] if review['decisions'][r['sha256']]['keep']]
FINAL=LOCAL/('추가_유사사진_%d장'%len(chosen));FINAL.mkdir(exist_ok=False)
cache=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file())
own=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file());add=0
for r in chosen:
    assert cache+add+r['bytes']<16*1024**3 and own+add+r['bytes']<2*1024**3
    assert shutil.disk_usage(ROOT).free-r['bytes']>30*1024**3
    raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
    p=FINAL/Path(r['path']).name
    with p.open('xb') as f:f.write(raw)
    add+=len(raw);r['final_path']=str(p);r['visual_review']=review['decisions'][r['sha256']]
with (FINAL/'선정목록_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['파일명','ID','품종','농장','비교샘플','유사강도','선정이유','공개근접계보주의','라벨치수일치','촬영일시','SHA256'])
    for r in chosen:w.writerow([Path(r['final_path']).name,r['image_id'],r['kind_type'],r['farm_id'],r['nearest_real']['file'],r['visual_review']['degree'],r['visual_review']['reason'],r['public_near_candidate'],r['dimension_match'],r['date_captured'],r['sha256']])
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14);COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images';boards=[]
for n in range(0,len(chosen),9):
    sub=chosen[n:n+9];imout=Image.new('RGB',(1080,3*290+40),'white');draw=ImageDraw.Draw(imout)
    draw.text((8,8),'최종 추가 유사사진 / 각 행 왼쪽 공개 실제 참조',font=font,fill='black')
    for j in range(3):
        block=sub[j*3:j*3+3]
        if not block:continue
        y=40+j*290;name=block[0]['nearest_real']['file']
        with Image.open(COMP/name) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((245,220));imout.paste(im,((270-im.width)//2,y))
        draw.text((5,y+225),'공개 '+name,font=font,fill='black')
        for i,r in enumerate(block,1):
            x=i*270
            with Image.open(r['final_path']) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((245,220));imout.paste(im,(x+(270-im.width)//2,y))
            draw.text((x+4,y+223),r['image_id']+' '+r['kind_type']+' '+r['farm_id'],font=font,fill='black')
            draw.text((x+4,y+245),r['visual_review']['degree']+' '+('계보주의' if r['public_near_candidate'] else ''),font=font,fill='black')
    p=LOCAL/('추가선별_공개비교_%02d_v1.jpg'%(n//9+1));imout.save(p,quality=92);boards.append(str(p))
result=dict(rows=chosen,count=len(chosen),folder=str(FINAL),boards=boards,kind_counts=dict(collections.Counter(r['kind_type'] for r in chosen)),farm_counts=dict(collections.Counter(r['farm_id'] for r in chosen)),public_near_count=sum(r['public_near_candidate'] for r in chosen),search_manifest=str(RUN/'scan_expanded_v2.json'),limits='개발농장 사전선정 일부. 전수 검색/독립 장면/동일 원본/학습 효용 미확인.')
with (RUN/'final_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
with (LOCAL/'먼저_읽기_v1.txt').open('x',encoding='utf-8') as f:f.write('추가_유사사진 폴더가 새 선별 원본 모음입니다. 기존49장은 별도 보존했습니다.\n선정목록 CSV에서 공개샘플·유사강도·선정이유·농장·품종을 볼 수 있습니다.\n공개근접계보주의는 동일사진 확인이 아니라 유사계보/근접hash 후보 주의입니다. 이 모음의 독립평가 이용은 별도 검증이 필요합니다. 원본 JPEG 무편집 복사.\n')
print(json.dumps({k:result[k] for k in ['count','folder','kind_counts','farm_counts','public_near_count']},ensure_ascii=False))
