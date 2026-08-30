#!/usr/bin/env python3
"""descarga open-perfectblend (mlabonne, 6 shards parquet) -> corpus_pb.bin
formato: {'conversations': [...], 'source': '...'} -> mismo ingest pares
contribuye ventana rotante extra al trainer nocturno"""
import urllib.request, os, random, sys

PB_BASE = "https://huggingface.co/datasets/mlabonne/open-perfectblend/resolve/main/data/train-0000{i}-of-00006.parquet"

def fetch(cfg=""):
    url = f"https://huggingface.co/api/datasets/mlabonne/open-perfectblend/parquet/default/train{cfg}"
    r = urllib.request.urlopen(url, timeout=60)
    return r.read().decode().strip(), r

def download_convert(shard=0):
    os.system(f"{sys.executable} -m pip install -q pyarrow 2>/dev/null")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow no disponible")
    url = PB_URL_SHARD(shard)
    req = urllib.request.Request(url, headers={"User-Agent": "tinyllama-adaptive/0.1"})
    print(f"descargando shard {shard} (~100-200MB)...")
    open("pb.parquet", "wb").write(urllib.request.urlopen(req, timeout=1200).read())
    print("parquet OK")
    tbl = pq.read_table("pb.parquet")
    col = None
    for c in ["conversations", "messages", "text"]:
        if c in tbl.column_names:
            col = c; break
    print("cols:", tbl.column_names, "| usando:", col)
    with open("pb.jsonl", "w", encoding="utf-8") as f:
        for i in range(tbl.num_rows):
            conv = tbl[col][i].as_py()
            if not conv: continue
            if isinstance(conv, str):
                f.write(json.dumps({"text": conv}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    os.remove("pb.parquet")

def PB_URL_SHARD(i):
    return f"https://huggingface.co/datasets/mlabonne/open-perfectblend/resolve/main/data/train-0000{i}-of-00006.parquet"

if __name__ == "__main__":
    import json
    OUT = "corpus_pb.bin"
    if not os.path.exists(OUT := "corpus_pb.bin") or os.path.getsize(OUT) < 1_000_000:
        if not os.path.exists("pb.jsonl"):
            download_convert(shard=int(os.environ.get("PB_SHARD", "0")))
        r = os.system("./ft_ingest corpus_pb.bin pb.jsonl")
        if r != 0:
            sys.exit(1)
        os.remove("pb.jsonl")
    data = open(OUT, "rb").read()
    seed = int(os.environ.get("GITHUB_RUN_NUMBER", "0") or 0)
    random.seed(1337 + seed % 997)
    target = 10_000_000
    start = random.randint(0, max(0, len(data) - target - 2))
    i = data.rfind(bytes([3]), 0, start)
    i = 0 if i == -1 else i
    j = data.find(bytes([3]), min(start + target, len(data) - 1))
    j = len(data) if j == -1 else j + 1
    open("corpus_pb_rot.bin", "wb").write(data[i:j])
    print(f"pb rot: {j-i} bytes desde {i} (total {len(data)})")