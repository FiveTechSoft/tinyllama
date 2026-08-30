"""descarga subsets de HF y los convierte a corpus.bin para el trainer"""
import json, urllib.request, sys, io

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tinyllama-adaptive/0.1"})
    return urllib.request.urlopen(req, timeout=120).read()

# ---- 1. alpaca-cleaned: streaming, primeros N registros ----
N_ALPACA = 8000
url = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json"
print("descargando alpaca-cleaned...")
data = json.loads(urllib.request.urlopen(url, timeout=180).read())
print("total alpaca:", len(data))
with open("hf_alpaca.jsonl", "w", encoding="utf-8") as f:
    for r in data[:N_ALPACA]:
        f.write(json.dumps({
            "instruction": r["instruction"],
            "input": r.get("input", ""),
            "output": r["output"],
        }, ensure_ascii=False) + "\n")
print("hf_alpaca.jsonl OK")

# ---- 2. TinyStories: streaming parquet->jsonl via HF datasets-server API ----
N_STORIES = 3000
print("descargando TinyStories (parquet index 0)...")
PQ_URL = "https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/default/train/0.parquet"
pq = fetch(PQ_URL)
print("parquet bytes:", len(pq))
# parquet->texto: extraemos strings largas legibles del binario (rapido y sin deps)
import re
text = pq.decode("latin-1")
cands = re.findall(r"[A-Za-z0-9 ,\.\!\?\'\"\n\r;:]{300,}", text)
stories = []
seen = set()
for s in cands:
    s2 = " ".join(s.split())
    if len(s2) > 300 and s2[:60] not in seen and "http" not in s2 and "{#" not in s2:
        seen.add(s2[:60])
        stories.append({"story": s2})
    if len(stories) >= N_STORIES:
        break
print("stories extraidas:", len(stories))
with open("hf_stories.jsonl", "w", encoding="utf-8") as f:
    for s in stories:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print("hf_stories.jsonl OK")