import sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pypdf import PdfReader

path = glob.glob(r"C:\Users\Anto\Downloads\Poor Lab*Qwen*.pdf")[0]
print("archivo:", path)
r = PdfReader(path)
print("paginas:", len(r.pages))
full = ""
for pg in r.pages:
    full += (pg.extract_text() or "") + "\n"
print("chars:", len(full))
open(r"C:\Users\Anto\AppData\Local\Temp\opencode\qwen_poor.txt", "w", encoding="utf-8").write(full)
print(full[:1200])