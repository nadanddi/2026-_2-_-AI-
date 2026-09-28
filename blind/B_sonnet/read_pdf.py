import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
from pypdf import PdfReader

reader = PdfReader(boot.PDF)
out_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\B_sonnet\pdf_text.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"num pages: {len(reader.pages)}\n")
    for i, page in enumerate(reader.pages):
        f.write(f"--- page {i+1} ---\n")
        f.write(page.extract_text() or "")
        f.write("\n")
print("done")
