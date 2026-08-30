#!/usr/bin/env python3
"""descarga FineTome-100k (parquet) -> corpus_ft_big.bin (corpus del trainer)
segun formato: slice rotante de ~15MB alineado a registros (0x03)"""
import urllib.request, os, random, sys

BIG = "corpus_ft_big.bin"

def download_convert():
    # 1. parquet -> jsonl (instala pyarrow si falta)
    os.system("pip install -q pyarrow 2>/dev/null || pip3 install -q pyarrow")
    import pyarrow.parquet as pq
    url = "https://huggingface.co/datasets/mlabonne/FineTome-100k/resolve/main/data/train-00000-of-00001.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "tinyllama-adaptive/0.1"})
    print("descargando parquet (~116MB)...")
    open("finetome.parquet", "wb").write(urllib.request.urlopen(req, timeout=600).read())
    print("parquet OK")
    tbl = pq.read_table("finetome.parquet", columns=["conversations"])
    print("rows:", tbl.num_rows)
    with open("hf_finetome_full.jsonl", "w", encoding="utf-8") as f:
        for i in range(tbl.num_rows):
            conv = tbl["conversations"][i].as_py()
            if conv:
                f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    os.remove("finetome.parquet")

def rot_slice():
    data = open(BIG, "rb").read()
    seed = int(os.environ.get("GITHUB_RUN_NUMBER", "0") or 0)
    random.seed(42 + seed % 1000)
    target = 15_000_000
    start = random.randint(0, max(0, len(data) - target - 2))
    i = data.rfind(bytes([3]), 0, start)
    i = 0 if i == -1 else i
    j = data.find(bytes([3]), min(start + target, len(data) - 1))
    j = len(data) if j == -1 else j + 1
    open("corpus_ft_rot.bin", "wb").write(data[i:j])
    print(f"rot slice: {j - i} bytes desde {i} (total {len(data)})")

if __name__ == "__main__":
    import json
    if not os.path.exists(BIG):
        if not os.path.exists("hf_finetome_full.jsonl"):
            download_convert()
        else:
            print("jsonl ya presente, omito descarga")
        # convierte con el compilado C
        r = os.system("./ft_ingest corpus_ft_big.bin hf_finetome_full.jsonl")
        if r != 0:
            sys.exit(1)
    rot_slice()