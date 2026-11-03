import json, re, hashlib, zipfile, posixpath
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
NS = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main', 'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}

def extract(version):
    path=ROOT/f'연구실/클로드/발표/덱/팜모니AI_발표자료_G05v{version}.pptx'
    result=[]
    with zipfile.ZipFile(path) as z:
        for i in range(1,34):
            slide=f'ppt/slides/slide{i}.xml'
            rel=ET.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
            links={r.attrib['Id']:posixpath.normpath(posixpath.join('ppt/slides',r.attrib['Target'])).lstrip('/') for r in rel}
            texts=[n.text or '' for n in ET.fromstring(z.read(slide)).findall('.//a:t',NS)]
            notes=[]; charts=[]
            for r in rel:
                p=links[r.attrib['Id']]
                if r.attrib['Type'].endswith('/notesSlide'):
                    notes=[n.text or '' for n in ET.fromstring(z.read(p)).findall('.//a:t',NS)]
                if r.attrib['Type'].endswith('/chart'):
                    c=ET.fromstring(z.read(p))
                    charts.append({'path':p,'values':[n.text for n in c.findall('.//c:numCache/c:pt/c:v',NS)],'texts':[n.text for n in c.findall('.//a:t',NS)],'categories':[n.text for n in c.findall('.//c:strCache/c:pt/c:v',NS)]})
            result.append({'slide':i,'texts':texts,'notes':notes,'charts':charts})
    (OUT/f'deck_v{version}_extracted.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def norm(s):
    return re.sub(r'\s+','',s.replace('**','').replace('`',''))

if __name__=='__main__':
    a,b=extract(7),extract(8)
    script=(ROOT/'연구실/클로드/발표/발표대본_G05v8_v4_2026-11-03.md').read_text(encoding='utf-8')
    sections={int(m.group(1)):m.group(2) for m in re.finditer(r'^## (\d+)\.[^\n]*\n(.*?)(?=^## |\Z)',script,re.M|re.S)}
    audit={'note_sync':[],'changed_screen':[],'chart_values':[],'hangul':0,'nonspace':0}
    for n in range(1,17):
        body='\n'.join(l for l in sections[n].splitlines() if not l.startswith('- ') and l.strip()!='---').strip()
        audit['hangul']+=len(re.findall('[가-힣]',body))
        audit['nonspace']+=len(re.sub(r'\s','',body))
        note='\n'.join(b[n-1]['notes'])
        m=re.search(r'\[말할 것\](.*?)(?=\[참고:|\[노트\]|\Z)',note,re.S)
        spoken=m.group(1).strip() if m else ''
        audit['note_sync'].append({'slide':n,'equal':norm(body)==norm(spoken),'script':body,'ppt_spoken':spoken})
    for i,(x,y) in enumerate(zip(a,b),1):
        if x['texts']!=y['texts']:audit['changed_screen'].append(i)
        for j,(c,d) in enumerate(zip(x['charts'],y['charts'])):
            old=[float(v) for v in c['values']];new=[float(v) for v in d['values']]
            audit['chart_values'].append({'slide':i,'chart':j,'equal':old==new,'percent_scaled':len(old)==len(new) and all(abs(v*100-w)<1e-8 for v,w in zip(old,new)),'percent_scaled_rounded_1dp':len(old)==len(new) and all(abs(v*100-w)<=.05000001 for v,w in zip(old,new)),'old':old,'new':new})
    audit['appendix_17_to_33_screen_identical']=all(a[i]['texts']==b[i]['texts'] for i in range(16,33))
    files=['연구실/클로드/발표/덱/팜모니AI_발표자료_G05v7.pptx','연구실/클로드/발표/덱/팜모니AI_발표자료_G05v8.pptx','연구실/클로드/발표/덱/build_g05_v8.js','연구실/클로드/발표/발표대본_G05v8_v4_2026-11-03.md']
    audit['hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (OUT/'comparison_v1.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='note_sync'},ensure_ascii=False,indent=2))
    print('NOTE SYNC',[(x['slide'],x['equal']) for x in audit['note_sync']])
