import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
t = open(r"C:\Users\Anto\AppData\Local\Temp\opencode\qwen_poor.txt", encoding="utf-8").read()

# encontrar el cuerpo de la seccion CMA (saltando el indice: tomar la ultima ocurrencia)
matches = list(re.finditer(r"(?i)curriculum model averaging", t))
print("ocurrencias CMA:", len(matches))
for m in matches[-2:]:
    chunk = " ".join(t[m.start():m.start()+900].split())
    print("==", chunk[:600], "\n")

# resultados concretos del abstract
m2 = re.search(r"(?i)approaches Qwen.{400,700}", t)
if m2:
    print("== resultados:", " ".join(m2.group(0).split())[:500])