import json, subprocess
out = subprocess.run(["gh", "run", "list", "--workflow=train_adaptive.yml", "--limit", "1", "--json", "databaseId"], capture_output=True, text=True)
run_id = json.loads(out.stdout)[0]["databaseId"]
log = subprocess.run(["gh", "run", "view", str(run_id), "--log"], capture_output=True, text=True)
for line in log.stdout.splitlines():
    l = line.lower()
    if any(k in l for k in ["ppl inicial", "ppl final", "publicado", "rechazado", "sin cambios", "commit", "corpus total"]):
        # solo la parte del mensaje, sin prefijos
        if "]Z " in line:
            msg = line.split("]Z ", 1)[1]
_ = None
# imprimir filtrado
for line in log.stdout.splitlines():
    if any(k in line.lower() for k in ["ppl inicial", "ppl final", "publicado", "rechazado", "sin cambios que publicar"]):
        parts = line.split("Z ", 1)
        if len(parts) == 2:
            print(parts[1][:160])