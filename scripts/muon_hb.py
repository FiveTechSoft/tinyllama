import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
t = open(r"C:\Users\Anto\AppData\Local\Temp\opencode\qwen_poor.txt", encoding="utf-8").read()

# la descripcion del update hyperball: buscar alrededor de la primera definicion
m = re.search(r"Hyperball.{3000}", t, re.I)
if m:
    print(" ".join(m.group(0).split())[:1800])