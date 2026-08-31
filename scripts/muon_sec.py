import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
t = open(r"C:\Users\Anto\AppData\Local\Temp\opencode\qwen_poor.txt", encoding="utf-8").read()

# seccion 3.3 Muon completa (buscar la ultima ocurrencia = cuerpo real)
idx = [m.start() for m in re.finditer(r"3\.3\.? Muon", t)]
print("ocurrencias 3.3 Muon:", len(idx))
if idx:
    chunk = t[idx[-1]:idx[-1]+3500]
    print(" ".join(chunk.split())[:2800])