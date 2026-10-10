from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,io,shutil
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
PUBLIC=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플/images'
d=json.loads((RUN/'final_manifest_v1.json').read_text(encoding='utf-8'));font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13);boards=[]
for n in range(0,len(d['rows']),9):
    canvas=Image.new('RGB',(1200,3*300+42),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,10),'각 쌍: 왼쪽 공개 샘플 / 오른쪽 선별 AI허브 원본',fill='black',font=font)
    for k,r in enumerate(d['rows'][n:n+9]):
        x=k%3*400;y=k//3*300+42;ref=r['visual_review']['reference']
        for offset,path in [(0,PUBLIC/ref),(200,Path(r['final_path']))]:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((190,225));canvas.paste(im,(x+offset+(200-im.width)//2,y))
        draw.line((x+399,y,x+399,y+295),fill='#cccccc')
        draw.text((x+3,y+228),'공개 '+ref,fill='black',font=font)
        draw.text((x+203,y+228),r['image_id']+' '+r['kind_type'],fill='black',font=font)
        draw.text((x+203,y+248),r['visual_review']['degree'],fill='black',font=font)
        draw.text((x+203,y+268),r['farm_id']+(' / 계보 주의' if r['public_near_candidate'] else ''),fill='black',font=font)
    b=io.BytesIO();canvas.save(b,format='JPEG',quality=92);raw=b.getvalue()
    total=sum(p.stat().st_size for task in (ROOT/'집/코덱스/local').glob('image_*') for p in task.rglob('*') if p.is_file());own=sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())
    assert total+len(raw)<16*1024**3 and own+len(raw)<2*1024**3 and shutil.disk_usage(ROOT).free-len(raw)>30*1024**3
    p=LOCAL/('공개샘플_사진별_짝비교_%02d_v2.jpg'%(n//9+1))
    with p.open('xb') as f:f.write(raw)
    boards.append(str(p))
with (RUN/'paired_boards_v2.json').open('x',encoding='utf-8') as f:json.dump(dict(boards=boards,each_pair_reference_matches_manual_review=True),f,ensure_ascii=False,indent=2)
print('paired boards',len(boards))
