from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from io import BytesIO
from lxml import etree as E
from openpyxl.utils.cell import range_boundaries,get_column_letter
import posixpath
P=Path(__file__).resolve().parent;src=P/'팜모니AI_발표자료_청자이해보완_v6.pptx';dst=P/'팜모니AI_발표자료_청자이해보완_v8.pptx'
C='http://schemas.openxmlformats.org/drawingml/2006/chart';S='http://schemas.openxmlformats.org/spreadsheetml/2006/main';ns={'c':C,'s':S};edits={}
with ZipFile(src) as z:
 for name in z.namelist():
  if not name.startswith('ppt/charts/chart') or not name.endswith('.xml'):continue
  root=E.fromstring(z.read(name));rname=posixpath.dirname(name)+'/_rels/'+posixpath.basename(name)+'.rels';rels=E.fromstring(z.read(rname));target=next(r.get('Target') for r in rels if r.get('Type','').endswith('/package'));emb=posixpath.normpath(posixpath.join(posixpath.dirname(name),target));patch={}
  for ref in root.findall('.//c:numRef',ns)+root.findall('.//c:strRef',ns):
   f=ref.findtext('c:f',namespaces=ns)
   cache=ref.find('c:numCache',ns);numeric=True
   if cache is None:cache=ref.find('c:strCache',ns);numeric=False
   if f is None or cache is None:continue
   _,rg=f.split('!',1);c1,r1,c2,r2=range_boundaries(rg.replace('$',''))
   for pt in cache.findall('c:pt',ns):
    idx=int(pt.get('idx'));v=pt.find('c:v',ns)
    if not numeric and name.endswith('/chart1.xml') and not (v.text or ''):v.text=str(idx)
    cell=f'{get_column_letter(c1+idx%(c2-c1+1))}{r1+idx//(c2-c1+1)}';patch[cell]=(numeric,v.text or '')
  if name.endswith('/chart1.xml'):
   for ax in root.findall('.//c:catAx',ns):
    skip=ax.find('c:tickLblSkip',ns)
    if skip is None:skip=E.SubElement(ax,'{'+C+'}tickLblSkip')
    skip.set('val','3')
  buf=BytesIO()
  with ZipFile(BytesIO(z.read(emb))) as wi,ZipFile(buf,'w',ZIP_DEFLATED) as wo:
   for info in wi.infolist():
    data=wi.read(info.filename)
    if info.filename=='xl/worksheets/sheet1.xml':
     wr=E.fromstring(data)
     for cell in wr.findall('.//s:c',ns):
      if cell.get('r') not in patch:continue
      num,val=patch[cell.get('r')]
      for child in list(cell):cell.remove(child)
      if num:cell.set('t','n');E.SubElement(cell,'{'+S+'}v').text=val
      else:cell.set('t','inlineStr');E.SubElement(E.SubElement(cell,'{'+S+'}is'),'{'+S+'}t').text=val
     have={c.get('r') for c in wr.findall('.//s:c',ns)}
     sd=wr.find('s:sheetData',ns)
     for address,(num,val) in patch.items():
      if address in have:continue
      rownum=''.join(c for c in address if c.isdigit())
      row=next((rr for rr in sd if rr.get('r')==rownum),None)
      if row is None:row=E.SubElement(sd,'{'+S+'}row',r=rownum)
      cc=E.SubElement(row,'{'+S+'}c',r=address,t='n' if num else 'inlineStr')
      if num:E.SubElement(cc,'{'+S+'}v').text=val
      else:E.SubElement(E.SubElement(cc,'{'+S+'}is'),'{'+S+'}t').text=val
     data=E.tostring(wr,encoding='UTF-8',xml_declaration=True,standalone=True)
    wo.writestr(info,data)
  edits[emb]=buf.getvalue();edits[name]=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
 assert not dst.exists()
 with ZipFile(dst,'w',ZIP_DEFLATED) as out:
  for info in z.infolist():out.writestr(info,edits.get(info.filename,z.read(info.filename)))
s=(P/'finalize_audience_v3.mjs').read_text(encoding='utf-8').replace('청자이해보완_v6.pptx','청자이해보완_v8.pptx').replace('finalizer_receipt_v3','finalizer_receipt_v5')
with (P/'finalize_audience_v5.mjs').open('x',encoding='utf-8') as f:f.write(s)
print('Restored exact source numeric literals in chart edit data; retained three-hour axis spacing.')
