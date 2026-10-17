from pathlib import Path
import json,hashlib
from pypdf import PdfReader
H=Path(__file__).resolve().parent;R=H.parents[3]
p=R/'공용/전문가_질의응답/2026-10-09_딸기_배지온도_EC_질문지.pdf'
r=PdfReader(p);txt='\n\n'.join('PAGE '+str(i+1)+'\n'+page.extract_text() for i,page in enumerate(r.pages))
(H/'전문가_질문지_추출_v1.txt').write_text(txt,encoding='utf-8')
print(txt)
