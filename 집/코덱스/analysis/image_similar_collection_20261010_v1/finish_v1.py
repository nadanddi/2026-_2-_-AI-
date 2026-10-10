from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json,csv,hashlib,io,collections,tarfile,sqlite3,shutil
from PIL import Image,ImageOps,ImageDraw,ImageFont
RUN=Path(__file__).resolve().parent; LOCAL=ROOT/'집/코덱스/local'/RUN.name
FINAL=LOCAL/'선별한_유사사진_49장';FINAL.mkdir(exist_ok=False)
d=json.loads((RUN/'manifest_v1.json').read_text(encoding='utf-8'))
exclude={
 'moke.jpg':{'435526':'검정 베드·가로 구도·흙 표면의 대응이 약함','417908':'검정 베드·가로 구도의 대응이 약함','601652':'검정 베드·가로 구도의 대응이 약함'},
 'wlid.jpg':{'416855':'넓은 잎 중심의 근접 구도가 약함'},
 'bphr.jpg':{'833519':'근접 잎 사진이며 열매 달린 수관 구도가 부족함','833559':'근접 잎 사진이며 열매 달린 수관 구도가 부족함'},
 'nqwq.jpg':{'415807':'검정 베드·조밀한 수관과 구도 대응이 약함'},
 'uexm.jpg':{'440869':'무성한 잎·꽃 수관보다 흙과 드문 잎이 두드러짐'}}
reasons={
 'jixx.jpg':'흰 재배대, 잎·꽃, 아래 자갈과 지지 구조 및 정면 구도',
 'ovvx.jpg':'흰 재배대·잎 수관·노란 표지; 열매와 잎 밀도 및 하단 배경 차이 있음',
 'qube.jpg':'흰 재배대의 흙, 길게 내려오는 줄기·잎, 자갈과 관수 구조',
 'wlid.jpg':'넓은 잎과 잎맥을 가까이 보는 구도; 배경과 노란 트랩 차이 있음',
 'xkdj.jpg':'잎이 겹치는 수관 구성; 공개 예시의 다양한 열매·꽃·넓은 구도와 차이 있음',
 'bpem.jpg':'잎·흰 꽃이 모인 전경; 합성 배경을 닮았다는 판정은 아님',
 'bphr.jpg':'잎·꽃 수관과 흰 재배대 전경; 열매·합성 배경 차이 있음',
 'yuag.jpg':'밝은 초록 잎과 줄기·꽃 전경; 합성 배경 차이 있음',
 'vfjx.jpg':'흰 화분·내려오는 줄기·금속 시설 배경 구도; 블러 효과는 비교 근거에서 제외',
 'nqwq.jpg':'잎·내려오는 꽃줄기 구성 일부; 검정 베드·표지·하단 배경 차이 있음, 잡음은 근거 제외',
 'jrtg.jpg':'잎·꽃 수관, 흰 재배대, 자갈·시설을 함께 담은 구도; 소금후추 잡음은 근거 제외',
 'kzbz.jpg':'흰 재배대·잎 수관·시설; 그림체보다 원본 장면 구성을 기준으로 비교',
 'nriv.jpg':'넓은 잎과 흰 꽃을 전면에 담은 흰 재배대 구도; 스타일 효과는 근거 제외',
 'ttmn.jpg':'잎·흰 꽃·줄기와 흰 재배대; 시점과 꽃 밀도 차이 있음',
 'ftkc.jpg':'흰 재배대·관수관·늘어진 줄기·잎 구성; 개별 식물 배치와 시점 차이 있음',
 'mobg.jpg':'잎·흰 꽃·늘어진 줄기, 흰 재배대와 자갈·지지대의 정면 구도',
 'uexm.jpg':'흰 재배대의 풍성한 잎·꽃 수관과 금속·자갈 배경; 동일 장면 여부 미확정',
 'dszm.jpg':'정면 잎·꽃 수관, 흰 재배대, 아래 자갈과 지지대 구도',
 'elue.jpg':'잎의 크기·겹침·잎맥을 보는 근접 구성; 꽃·배경 범위 차이 있음',
 'xeeo.jpg':'잎·줄기와 흰 화분·금속 시설 구도; 식물 위치와 잎 밀도 차이 있음'}
strong={'jixx.jpg','qube.jpg','vfjx.jpg','mobg.jpg','dszm.jpg'}
decisions=[];selected=[]
for r in d['rows']:
    kept=[]
    for z in r['references']:
        reject=exclude.get(z['file'],{}).get(r['image_id'])
        a=dict(image_id=r['image_id'],public_file=z['file'],public_type=z['type'],decision='제외' if reject else '유지',degree='제외' if reject else ('장면 유사' if z['file'] in strong else '구성 요소 유사'),reason=reject or reasons[z['file']])
        decisions.append(a)
        if not reject:kept.append(a)
    if not kept:continue
    raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
    p=FINAL/Path(r['path']).name
    with p.open('xb') as f:f.write(raw)
    selected.append(dict(r,final_path=str(p),visual_references=kept,status='육안 확인 후 유지'))
assert len(selected)==49
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15);boards=[]
for start in range(0,len(selected),12):
    sub=selected[start:start+12];canvas=Image.new('RGB',(960,((len(sub)+3)//4)*275+40),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),'선별한 AI허브 유사 원본 — %d / 49'%(start+1),font=font,fill='black')
    for i,r in enumerate(sub):
        x=i%4*240;y=i//4*275+40
        with Image.open(r['final_path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((205,220));canvas.paste(im,(x+(240-im.width)//2,y))
        draw.text((x+5,y+222),r['image_id']+' '+r['kind'],font=font,fill='black')
        draw.text((x+5,y+244),', '.join(z['public_file'].removesuffix('.jpg') for z in r['visual_references']),font=font,fill='black')
    p=LOCAL/('선별사진_미리보기_%02d_v1.jpg'%(start//12+1));canvas.save(p,quality=92);boards.append(str(p))
with (RUN/'육안_선정판정_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
manifest=dict(search_scope=d['scope'],images=selected,reference_decisions=decisions,boards=boards,count=len(selected),kind_counts=dict(collections.Counter(r['kind'] for r in selected)),farm_counts=dict(collections.Counter(r['farm'] for r in selected)),prefix_counts=dict(collections.Counter(r['prefix'] for r in selected)),limits='캐시1000의 상위 후보에서 육안 선별. 동일 원본/독립 장면/학습 효용 확인 아님. 공개 샘플로 학습 성능을 검증하지 않음.')
with (RUN/'final_manifest_v1.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
with (FINAL/'선정목록_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['파일명','ID','품종','농장','그룹접두사','촬영일시','비교공개샘플','선정이유','SHA256'])
    for r in selected:w.writerow([Path(r['final_path']).name,r['image_id'],r['kind'],r['farm'],r['prefix'],r['date'],'; '.join(z['public_file'] for z in r['visual_references']),'; '.join(z['reason'] for z in r['visual_references']),r['sha256']])
with (LOCAL/'먼저_읽기_v1.txt').open('x',encoding='utf-8') as f:
    f.write('선별한_유사사진_49장 폴더: 편집하지 않은 AI허브 JPEG 원본49장과 선정목록 CSV.\n선별사진_미리보기_01~05: 최종 모음 미리보기.\n*_공개샘플_비교판: 자동후보 단계 비교판. 제외된 후보도 있으므로 육안_선정판정 CSV와 구분할 것.\n금실%d/설향%d. 검색은 VS금실2·설향3 캐시1000에 한정. 사진 수는 독립 식물/장면 수가 아님. 공개 유형의 편집 효과를 닮은 원본이라는 뜻이 아니라 작물·시설·구도 비교. 전체 생성 moke에 충분히 닮은 후보는 이번 모음에서 찾지 못함.\n'% (manifest['kind_counts'].get('금실',0),manifest['kind_counts'].get('설향',0)))
print(json.dumps({k:manifest[k] for k in ['count','kind_counts','farm_counts','prefix_counts']},ensure_ascii=False))
