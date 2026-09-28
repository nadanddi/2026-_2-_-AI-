import sys; sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot, pypdf
r = pypdf.PdfReader(boot.PDF)
open("pdf.txt","w",encoding="utf-8").write("\n\n=====PAGE=====\n".join(p.extract_text() for p in r.pages))
