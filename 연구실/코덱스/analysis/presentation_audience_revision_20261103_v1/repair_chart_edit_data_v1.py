"""Repair inherited chart workbook links from existing literal caches; no data invention."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from io import BytesIO
from lxml import etree as E
from openpyxl import Workbook
from openpyxl.utils.cell import range_boundaries
import posixpath,json
P=Path(__file__).resolve().parent
NS={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/package/2006/relationships'}
C=NS['c']; src=P/'팜모니AI_발표자료_청자이해보완_v5.pptx'; dst=P/'팜모니AI_발표자료_청자이해보완_v6.pptx'
edits={};logs=[]
with ZipFile(src) as z:
 for name in z.namelist():
  if not name.startswith('ppt/charts/chart') or not name.endswith('.xml'):continue
  r=E.fromstring(z.read(name));wb=Workbook();wb.remove(wb.active)
  for ser in r.findall('.//c:ser',NS):
   for kind in ['tx','cat','val']:
    root=ser.find('c:'+kind,NS)
    if root is None:continue
    mult=root.find('c:multiLvlStrRef',NS)
    if mult is not None:
     cache=mult.find('c:multiLvlStrCache',NS);levels=cache.findall('c:lvl',NS)
     assert len(levels)==1,'Do not flatten true hierarchical categories'
     nr=E.Element('{'+C+'}strRef');nr.append(E.fromstring(E.tostring(mult.find('c:f',NS))))
     nc=E.SubElement(nr,'{'+C+'}strCache');nc.append(E.fromstring(E.tostring(cache.find('c:ptCount',NS))))
     for pt in levels[0]:nc.append(E.fromstring(E.tostring(pt)))
     root.remove(mult);root.append(nr)
    for ref in list(root):
     if not ref.tag.endswith('Ref'):continue
     formula=ref.find('c:f',NS);cache=ref.find('c:strCache',NS)
     numeric=False
     if cache is None:cache=ref.find('c:numCache',NS);numeric=True
     if formula is None or cache is None:continue
     sheet,rg=formula.text.split('!',1);sheet=sheet.strip("'")
     if sheet not in wb.sheetnames:wb.create_sheet(sheet)
     ws=wb[sheet];c1,r1,c2,r2=range_boundaries(rg.replace('$',''))
     pts={int(pt.get('idx')):pt.findtext('c:v',default='',namespaces=NS) for pt in cache.findall('c:pt',NS)}
     count=int(cache.find('c:ptCount',NS).get('val'))
     assert count==(c2-c1+1)*(r2-r1+1)
     for idx in range(count):
      val=pts.get(idx,''); rr=r1+idx//(c2-c1+1);cc=c1+idx%(c2-c1+1)
      ws.cell(rr,cc,float(val) if numeric and val else (val if val else None))
  relpath=posixpath.dirname(name)+'/_rels/'+posixpath.basename(name)+'.rels'
  rel=E.fromstring(z.read(relpath));targets=[n.get('Target') for n in rel if n.get('Type','').endswith('/package')]
  assert len(targets)==1
  emb=posixpath.normpath(posixpath.join(posixpath.dirname(name),targets[0]))
  buf=BytesIO();wb.save(buf);edits[emb]=buf.getvalue();edits[name]=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
  logs.append({'chart':name,'workbook':emb,'repair':'rebuild editable sheet solely from unchanged chart literal caches; flatten a single-level category wrapper'})
 assert not dst.exists()
 with ZipFile(dst,'w',ZIP_DEFLATED) as out:
  for info in z.infolist():out.writestr(info,edits.get(info.filename,z.read(info.filename)))
with (P/'chart_edit_data_repair_v1.json').open('x',encoding='utf-8') as f:json.dump(logs,f,ensure_ascii=False,indent=2)
s=(P/'finalize_audience_v2.mjs').read_text(encoding='utf-8').replace('청자이해보완_v5.pptx','청자이해보완_v6.pptx').replace('finalizer_receipt_v2','finalizer_receipt_v3')
with (P/'finalize_audience_v3.mjs').open('x',encoding='utf-8') as f:f.write(s)
print('Repaired chart edit workbooks from original chart values:',len(logs))
