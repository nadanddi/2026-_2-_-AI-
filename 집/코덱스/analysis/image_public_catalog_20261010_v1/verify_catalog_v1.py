from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,hashlib,sqlite3,base64,io,collections
from html.parser import HTMLParser
from PIL import Image
from pypdf import PdfReader
RUN=Path(__file__).resolve().parent
cat=json.loads((RUN/'catalog_v1.json').read_text(encoding='utf-8'));rows=cat['rows']
labels=list(csv.DictReader(Path(cat['sources']['CSV']).open(encoding='utf-8-sig')))
table=list(csv.DictReader((RUN/'공개21_개별정리_v1.csv').open(encoding='utf-8-sig')))
assert len(rows)==len(table)==len(labels)==21
assert [(r['file'],r['label'],r['type']) for r in rows]==[(r['file'],int(r['label']),r['type']) for r in labels]
assert [r['file'] for r in table]==[r['file'] for r in rows]
for row,record in zip(rows,table):assert all(str(v)==record[k] for k,v in row.items())
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.images=[];self.ids=[];self.script=False;self.payload='';self.buttons=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img' and a.get('src','').startswith('data:'):self.images.append(a['src'])
        if tag=='article':self.ids.append(a['data-id'])
        if tag=='script' and a.get('id')=='catalog-data':self.script=True
        if tag=='button' and 'tab' in a.get('class','').split():self.buttons.append(a['data-type'])
    def handle_endtag(self,tag):
        if tag=='script':self.script=False
    def handle_data(self,data):
        if self.script:self.payload+=data
parser=Parser();parser.feed((RUN/'공개21_이미지목록_v1.html').read_text(encoding='utf-8'))
assert parser.ids==[r['file'] for r in rows] and len(parser.images)==21 and json.loads(parser.payload)==rows
assert parser.buttons==['all']+list(cat['types'])
pdf=PdfReader(cat['sources']['PDF']);pages=[''.join(p.extract_text().split()) for p in pdf.pages]
notes=(RUN/'유형별_사진별_정리_v1.txt').read_text(encoding='utf-8')
for row,encoded in zip(rows,parser.images):
    raw=Path(row['source_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
    assert raw==base64.b64decode(encoded.split(',',1)[1])
    with Image.open(io.BytesIO(raw)) as im:im.verify()
    with Image.open(io.BytesIO(raw)) as im:assert list(im.size)==[row['width'],row['height']]
    assert row['file'] in pages[row['PDF_page']-1] and row['file'] in notes and row['official_summary'] in notes and row['direct_visual_observation'] in notes
conn=sqlite3.connect(':memory:');conn.execute('create table samples(file text,label int,type text,w int,h int)');conn.executemany('insert into samples values(?,?,?,?,?)',[(r['file'],r['label'],r['type'],r['width'],r['height']) for r in rows])
counts=dict(conn.execute('select type,count(*) from samples group by type'));labs={str(k):v for k,v in conn.execute('select label,count(*) from samples group by label')}
dims={f'{w}x{h}':n for w,h,n in conn.execute('select w,h,count(*) from samples group by w,h')}
assert counts==cat['counts'] and labs=={'0':3,'1':18} and sorted(dims.values())==[4,17]
boards=json.loads((RUN/'visual_boards_v1.json').read_text(encoding='utf-8'));assert len(boards)==7
for board in boards:
    p=Path(board['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==board['sha256']
    with Image.open(p) as im:assert im.size==(1240,785)
assert hashlib.sha256(Path(cat['sources']['PDF']).read_bytes()).hexdigest()==cat['sources']['PDF_sha256']
assert hashlib.sha256(Path(cat['sources']['CSV']).read_bytes()).hexdigest()==cat['sources']['CSV_sha256']
result={'all_checks_pass':True,'fresh_sources':21,'PDF_pages':len(pdf.pages),'independent_SQL_type_counts':counts,'independent_SQL_label_counts':labs,'independent_SQL_dimensions':dims,'CSV_JSON_HTML_TXT_order_and_content_match':True,'embedded21_original_JPEG_bytes_match':True,'seven_boards_SHA_match':True,'runtime_browser_interaction':'not tested: local file protocol blocked by browser URL policy; no workaround attempted','scope':'catalog content/hash/structure verification; not model-performance or edited-region detection'}
with (RUN/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
