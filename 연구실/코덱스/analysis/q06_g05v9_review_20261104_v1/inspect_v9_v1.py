import json,re,hashlib,zipfile,posixpath
from pathlib import Path
from xml.etree import ElementTree as E
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
def read(v,count):
    out=[]
    with zipfile.ZipFile(ROOT/f'연구실/클로드/발표/덱/팜모니AI_발표자료_G05v{v}.pptx') as z:
        for i in range(1,count+1):
            r=E.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            notes=[];charts=[]
            for x in r:
                p=posixpath.normpath(posixpath.join('ppt/slides',x.attrib['Target'])).lstrip('/')
                if x.attrib['Type'].endswith('/notesSlide'):notes=[n.text or '' for n in E.fromstring(z.read(p)).findall('.//a:t',NS)]
                if x.attrib['Type'].endswith('/chart'):
                    c=E.fromstring(z.read(p));charts.append({'values':[n.text for n in c.findall('.//c:numCache/c:pt/c:v',NS)],'labels':[n.text for n in c.findall('.//c:strCache/c:pt/c:v',NS)]})
            out.append({'slide':i,'texts':[n.text or '' for n in E.fromstring(z.read(f'ppt/slides/slide{i}.xml')).findall('.//a:t',NS)],'notes':notes,'charts':charts})
    (OUT/f'deck_v{v}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out
def norm(s):return re.sub(r'\s+','',s.replace('**','').replace('`',''))
a,b=read(8,33),read(9,34)
s=(ROOT/'연구실/클로드/발표/발표대본_G05v9_v5_2026-11-04.md').read_text(encoding='utf-8')
parts={int(m.group(1)):(m.group(2),m.group(3)) for m in re.finditer(r'^## (\d+)\.([^\n]*)\n(.*?)(?=^## |\Z)',s,re.M|re.S)}
sync=[];times=[];syllables=[]
for n in range(1,18):
    title,raw=parts[n];body='\n'.join(l for l in raw.splitlines() if not l.startswith('- ') and l.strip()!='---').strip()
    notes='\n'.join(t for t in b[n-1]['notes'] if t!=str(n));m=re.search(r'\[말할 것\](.*?)(?=\[노트\]|\[화면|\[참고:|\Z)',notes,re.S)
    sync.append({'slide':n,'equal':norm(body)==norm(m.group(1) if m else ''),'script':body,'note':m.group(1) if m else ''})
    t=re.search(r'\((\d+):(\d+)\)',title);times.append(int(t[1])*60+int(t[2]))
    syllables.append(len(re.findall('[가-힣]',body)))
chart=[]
for old,new in [(6,7),(8,9),(9,10),(13,14),(14,15)]:
    chart.append({'old':old,'new':new,'equal':[x['values'] for x in a[old-1]['charts']]==[x['values'] for x in b[new-1]['charts']]})
files=['연구실/클로드/발표/덱/팜모니AI_발표자료_G05v8.pptx','연구실/클로드/발표/덱/팜모니AI_발표자료_G05v9.pptx','연구실/클로드/발표/덱/build_g05_v9.js','연구실/클로드/발표/발표대본_G05v9_v5_2026-11-04.md']
def without_page(slide):
    t=list(slide['texts'])
    if t and t[-1]==str(slide['slide']):t.pop()
    return t
audit={'sync':sync,'syllables':syllables,'hangul':sum(syllables),'times':times,'seconds':sum(times),'chart_comparison':chart,'appendix_equal_excluding_page_number':all(without_page(a[i])==without_page(b[i+1]) for i in range(16,33)),'old_dialogue_slides':[x['slide'] for x in b[:17] if 'G05 v7까지의 대사' in '\n'.join(x['notes']) or '발표에서 읽지 않음' in '\n'.join(x['notes'])],'hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
(OUT/'audit_v1.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='sync'},ensure_ascii=False,indent=2))
print('SYNC',[(x['slide'],x['equal']) for x in sync])
