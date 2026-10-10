import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,io,hashlib,base64,html,collections,textwrap
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
RUN=Path(__file__).resolve().parent;LOCAL=ROOT/'집/코덱스/local'/RUN.name
LOCAL.mkdir(parents=True,exist_ok=True)
COMP=ROOT/'공용/대회자료/이미지데이터/참가자_배포/공개샘플'
PDF=COMP.parent/'공개샘플21_설명.pdf';LABELS=COMP/'sample_labels.csv'
notes=json.loads((RUN/'curated_notes_v1.json').read_text(encoding='utf-8'))
labels=list(csv.DictReader(LABELS.open(encoding='utf-8-sig')))
assert len(labels)==21 and len({r['file'] for r in labels})==21
assert set(notes['files'])=={r['file'] for r in labels}
reader=PdfReader(PDF);assert len(reader.pages)==3
pages=[p.extract_text() for p in reader.pages]
with (RUN/'PDF_text_fresh_v1.txt').open('x',encoding='utf-8') as f:
    for i,text in enumerate(pages,1):f.write(f'\n=== PDF {i}쪽 ===\n{text}\n')
rows=[];dataurls={};images={}
for n,lab in enumerate(labels,1):
    p=COMP/'images'/lab['file'];raw=p.read_bytes();typ=notes[lab['type']];detail=notes['files'][lab['file']]
    with Image.open(io.BytesIO(raw)) as im:im.load();images[lab['file']]=im.convert('RGB').copy();size=im.size;fmt=im.format
    assert lab['file'] in ''.join(pages[typ['page']-1].split())
    rows.append(dict(order=n,file=lab['file'],label=int(lab['label']),type=lab['type'],type_ko=typ['name'],PDF_page=typ['page'],official_summary=detail['official'],direct_visual_observation=detail['visual'],unknown=typ['unknown'],AIhub_selection_reference=detail['selection'],augmentation_definition=typ['recipe'],source_path=str(p),width=size[0],height=size[1],format=fmt,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),observation_scope='21개 파일을 각 원래 해상도로 직접 확인; 정확한 편집 위치 추정 없음'))
    dataurls[lab['file']]='data:image/jpeg;base64,'+base64.b64encode(raw).decode()
assert collections.Counter(r['type'] for r in rows)=={t:3 for t in ['real','AI_gen','composite','filter','full_style','partial_gen','partial_style']}
catalog={'sources':{'PDF':str(PDF),'PDF_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'CSV':str(LABELS),'CSV_sha256':hashlib.sha256(LABELS.read_bytes()).hexdigest()},'types':{k:v for k,v in notes.items() if k!='files'},'rows':rows,'counts':dict(collections.Counter(r['type'] for r in rows)),'label_counts':dict(collections.Counter(r['label'] for r in rows)),'scope':'공개 샘플 설명·참고 카탈로그; 모델학습/성능/증강실행 없음; 원본 사진/설명서 불변'}
with (RUN/'catalog_v1.json').open('x',encoding='utf-8') as f:json.dump(catalog,f,ensure_ascii=False,indent=2)
with (RUN/'공개21_개별정리_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
order=list(catalog['types'])
titlefont=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',25);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
boards=[]
for typ in order:
    group=[r for r in rows if r['type']==typ];name=catalog['types'][typ]['name'];canvas=Image.new('RGB',(1240,785),'#f3f5f2');draw=ImageDraw.Draw(canvas)
    draw.text((24,18),name+' · 각 3장 · 라벨 '+str(group[0]['label']),font=titlefont,fill='#203c30')
    for i,r in enumerate(group):
        x=20+i*408;im=images[r['file']].copy();im.thumbnail((390,520),Image.Resampling.LANCZOS)
        draw.rounded_rectangle((x,65,x+394,775),radius=9,fill='white')
        canvas.paste(im,(x+(394-im.width)//2,70+(520-im.height)//2))
        draw.text((x+10,605),f"{r['file']} · {r['width']}×{r['height']} · PDF {r['PDF_page']}쪽",font=font,fill='#203c30')
        lines=textwrap.wrap(r['official_summary'],width=24)
        for j,line in enumerate(lines):draw.text((x+10,633+j*23),line,font=font,fill='#444444')
        assert 633+len(lines)*23<775
    target=LOCAL/f'{typ}_유형판_v1.jpg'
    with target.open('xb') as f:canvas.save(f,format='JPEG',quality=94,subsampling=2)
    boards.append({'type':typ,'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
with (RUN/'visual_boards_v1.json').open('x',encoding='utf-8') as f:json.dump(boards,f,ensure_ascii=False,indent=2)
header='공개샘플 21장 정리 · 2026-10-10 집 코덱스\n공식 설명 PDF 3쪽·라벨 CSV·각 사진 전량 대조. 실제 3장, 양성 6유형×3장=18장.\n사진은 원본 그대로이며, 관찰 특징은 정답을 결정하는 단일 규칙이 아니다.\n원본/편집 마스크가 없는 부분 생성·부분 스타일은 편집된 위치를 지정하지 않는다.\n\n'
lines=[header]
for typ in order:
    d=catalog['types'][typ];lines.extend([f"\n[{d['name']}] · PDF {d['page']}쪽\n",d['definition']+'\n',f"확인 불가: {d['unknown']}\n",f"후속 증강 방식: {d['recipe']}\n"])
    for r in rows:
        if r['type']==typ:lines.append(f"\n{r['order']:02}. {r['file']} | 라벨 {r['label']} | {r['width']}×{r['height']}\n  공식 설명 요약: {r['official_summary']}\n  사진 직접 관찰: {r['direct_visual_observation']}\n  AI허브 선정 참고: {r['AIhub_selection_reference']}\n")
lines.extend(['\n공통 해석\n','전체 생성은 장면 자체 생성, 전체 스타일은 사진 바탕 표현 양식 변경이다.\n','지정 필터 3장의 정확한 종류는 알려졌지만 강도·모델·시드까지 복원한 것은 아니다. 의도 JPEG 압축률 변경도 범위에 있다.\n','배포용 공통 크롭·크기 조정·JPEG 저장은 기존 라벨을 바꾸지 않는다. 자연적인 초점 흐림·잡음만으로 양성 판정하지 않는다.\n','파일명·여백·해상도·노란 표지·흐림은 모든 이미지에 통하는 정답 규칙이 아니다.\n','이 21장의 유형 비율과 정확도로 비공개 평가의 분포·성능을 판단하지 않는다.\n','이번 요청의 우선 작업은 카탈로그 정리다. 유사원본 100장 제작 선정·필터 300개·추가 생성은 아직 실행하지 않았다. 기존 유사원본 검색과 첫 생성 2파일 기록은 별도 보존했다.\n'])
with (RUN/'유형별_사진별_정리_v1.txt').open('x',encoding='utf-8') as f:f.write(''.join(lines))
cards=[]
for r in rows:
    esc=lambda s:html.escape(str(s),quote=True)
    fields=''.join(f'<dt>{k}</dt><dd>{esc(r[v])}</dd>' for k,v in [('공식 설명','official_summary'),('사진 직접 관찰','direct_visual_observation'),('확인 불가','unknown'),('AI허브 선정 참고','AIhub_selection_reference'),('후속 증강 방식','augmentation_definition')])
    cards.append(f'<article class="card" data-type="{r["type"]}" data-id="{r["file"]}"><button class="picture" data-id="{r["file"]}" aria-label="{r["file"]} 원본 확대"><img src="{dataurls[r["file"]]}" alt="{esc(r["file"]+" "+r["type_ko"])}"></button><div class="content"><span class="badge">{r["type_ko"]} · 라벨 {r["label"]}</span><h3>{r["file"]}</h3><p class="meta">{r["width"]} × {r["height"]} · 설명 PDF {r["PDF_page"]}쪽</p><dl>{fields}</dl></div></article>')
options='<button class="tab selected" data-type="all">전체 21장</button>'+''.join(f'<button class="tab" data-type="{typ}">{catalog["types"][typ]["name"]} 3장</button>' for typ in order)
payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 샘플 21장 · 유형별 정리</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f2;color:#22342a;font-family:"Malgun Gothic",sans-serif;line-height:1.65}header,main{max-width:1380px;margin:auto;padding:28px}h1{font-size:30px;margin:0}header p{max-width:1060px;margin:10px 0}.notice{padding:14px 18px;background:#fff;border-left:4px solid #628868;border-radius:6px;font-size:14px}.toolbar{position:sticky;top:0;z-index:5;background:#f3f5f2ee;backdrop-filter:blur(8px);padding:12px 28px;border-bottom:1px solid #d8e0d7}.tabs{max-width:1324px;margin:auto;display:flex;gap:8px;flex-wrap:wrap}.tab{border:1px solid #bccbbb;background:#fff;padding:9px 12px;border-radius:7px;color:#22342a;font:inherit;font-size:13px;cursor:pointer}.selected{background:#315b3d;color:white;border-color:#315b3d}.count{max-width:1324px;margin:8px auto 0;font-size:13px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}.card{background:white;border:1px solid #d8e0d7;border-radius:12px;overflow:hidden}.card[hidden]{display:none}.picture{width:100%;height:410px;background:#e8ece5;border:0;padding:12px;cursor:zoom-in}.picture img{width:100%;height:100%;object-fit:contain}.content{padding:20px}h3{margin:8px 0 0;font-size:22px}.badge{font-size:12px;background:#e7efe3;padding:5px 9px;border-radius:5px}.meta{font-size:13px;color:#657263;margin:2px 0 16px}dl{font-size:13px;margin:0}dt{font-weight:bold;margin-top:12px}dd{margin:3px 0;color:#415140}dialog{border:0;border-radius:10px;width:94vw;max-width:1500px;height:94vh;background:#f3f5f2;padding:18px}dialog::backdrop{background:#000b}.modalbar{display:flex;justify-content:space-between;align-items:center}.modalbody{height:calc(100% - 55px);display:grid;grid-template-columns:2fr 1fr;gap:22px;padding-top:12px}.modalbody img{width:100%;height:100%;object-fit:contain;background:#e8ece5}.modalinfo{overflow:auto;font-size:14px}.close{padding:8px 16px;cursor:pointer}footer{max-width:1324px;margin:12px auto 30px;font-size:13px;color:#657263}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.modalbody{grid-template-columns:1fr}.modalbody img{height:65vh}}@media(max-width:620px){.grid{grid-template-columns:1fr}header,main{padding:20px}.toolbar{padding:12px 20px}}@media print{.toolbar{position:static}.picture{height:330px}.card{break-inside:avoid}.grid{grid-template-columns:repeat(3,1fr)}}
</style><header><h1>공개 샘플 21장 · 유형별 정리</h1><p>실제 촬영 3장 + 여섯 생성·편집 유형 각 3장. 설명 PDF 3쪽, 공식 라벨 CSV와 실제 사진을 대조했습니다. 사진을 누르면 원본 해상도로 확대됩니다.</p><div class="notice">공식 설명 · 사진 직접 관찰 · 확인 불가 정보를 구분했습니다. 부분 생성·부분 스타일의 편집 전 원본과 마스크가 제공되지 않아 정확한 편집 위치를 표시하지 않습니다. 관찰 특징 하나를 정답 규칙으로 사용하지 않습니다.</div></header><nav class="toolbar"><div class="tabs">OPTIONS</div><p class="count" id="count">21장 표시 중</p></nav><main><div class="grid">CARDS</div></main><footer>출처: 공용/대회자료/이미지데이터/참가자_배포/공개샘플21_설명.pdf 및 공개샘플/sample_labels.csv · 원본 사진 수정 없음 · 제작 연산의 종류와 미제공 세부를 구분 · 2026-10-10 집 코덱스</footer><dialog id="zoom"><div class="modalbar"><strong id="zoomtitle"></strong><button class="close" id="close">닫기</button></div><div class="modalbody"><img id="zoomimg" alt=""><div class="modalinfo" id="zoominfo"></div></div></dialog><script type="application/json" id="catalog-data">PAYLOAD</script><script>
const rows=JSON.parse(document.querySelector('#catalog-data').textContent);const dialog=document.querySelector('#zoom');document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('selected',x===b));let count=0;document.querySelectorAll('.card').forEach(c=>{c.hidden=b.dataset.type!=='all'&&c.dataset.type!==b.dataset.type;if(!c.hidden)count++});document.querySelector('#count').textContent=count+'장 표시 중'});document.querySelectorAll('.picture').forEach(b=>b.onclick=()=>{const r=rows.find(x=>x.file===b.dataset.id);document.querySelector('#zoomtitle').textContent=r.file+' · '+r.type_ko+' · 라벨 '+r.label;const img=document.querySelector('#zoomimg');img.src=b.querySelector('img').src;img.alt=r.file;const info=document.querySelector('#zoominfo');info.innerHTML='';for(const [title,key]of [['공식 설명','official_summary'],['사진 직접 관찰','direct_visual_observation'],['확인 불가','unknown'],['AI허브 선정 참고','AIhub_selection_reference'],['후속 증강 방식','augmentation_definition']]){const h=document.createElement('h4');h.textContent=title;const p=document.createElement('p');p.textContent=r[key];info.append(h,p)}dialog.showModal()});document.querySelector('#close').onclick=()=>dialog.close();dialog.onclick=e=>{if(e.target===dialog)dialog.close()};
</script></html>'''.replace('OPTIONS',options).replace('CARDS',''.join(cards)).replace('PAYLOAD',payload)
with (RUN/'공개21_이미지목록_v1.html').open('x',encoding='utf-8') as f:f.write(page)
assert sum(p.stat().st_size for p in LOCAL.rglob('*') if p.is_file())<80*1024**2
print(json.dumps({'rows':len(rows),'counts':catalog['counts'],'label_counts':catalog['label_counts'],'boards':len(boards),'html_bytes':len(page.encode('utf-8'))},ensure_ascii=False))
