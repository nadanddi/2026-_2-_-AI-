import sys,json,io
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
from PIL import Image,ImageOps,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집'/'코덱스'/'local'/'image_similarity_20261009_v1'/'핵심비교_v1.jpg'
if OUT.exists(): raise FileExistsError(OUT)
COMP=ROOT/'공용'/'대회자료'/'이미지데이터'/'참가자_배포'/'공개샘플'/'images'
rows=[r for r in json.loads((HERE/'comparison_v1.json').read_text(encoding='utf-8'))['per_image'] if r['label']==0 or r['file']=='uexm.jpg']
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
canvas=Image.new('RGB',(1200,1375),'#f7f8fa'); d=ImageDraw.Draw(canvas)
for col,title in enumerate(['대회 공개 샘플','AI허브 · 형태 기준 최근접','AI허브 · 색상 기준 최근접']):d.text((col*400+12,12),title,font=font,fill='#17243b')
with ZipFile('G:/내 드라이브/농업 AI 경진대회/이미지 미션/New_Sample.zip의 사본') as z:
 for row,q in enumerate(rows):
  picks=[q['file'],q['phash'][0]['name'],q['color'][0]['name']]
  for col,n in enumerate(picks):
   raw=(COMP/n).read_bytes() if col==0 else z.read(n)
   with Image.open(io.BytesIO(raw)) as im:
    t=ImageOps.exif_transpose(im).convert('RGB'); t.thumbnail((380,280))
   x=col*400; y=48+row*330
   canvas.paste(t,(x+(400-t.width)//2,y)); d.text((x+12,y+287),(Path(n).name+(' · 실제' if q['label']==0 else ' · 부분생성')) if col==0 else Path(n).name,font=small,fill='#17243b')
   if col==1:d.text((x+12,y+309),f"형태 해시 거리 {q['phash'][0]['distance']:.0f} (낮을수록 가까움)",font=small,fill='#586575')
   if col==2:d.text((x+12,y+309),f"색상 거리 {q['color'][0]['distance']:.3f} (유사도 % 아님)",font=small,fill='#586575')
canvas.save(OUT,quality=94)
print(OUT)
