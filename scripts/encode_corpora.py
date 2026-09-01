"""codifica TODOS los corpus a corpus_ids.bin (formato: u32 n + u32 ids[])
con el encoder greedy python (validado roundtrip), ?????????????? vocab.bpe
tambien genera corpus_hf_ids, corpus_peng_ids, corpus_st_ids, corpus_ft_ids
"""
import struct

def load_vocab(path=r"adaptive\vocab.bpe"):
    toks = {}
    for l in open(path, encoding="utf-8").read().splitlines():
        p = l.split("\t")
        if len(p) >= 3:
            toks[int(p[0])] = p[1] + p[2].replace("\r", "")
    # greedy: array ordenado de (texto, id) por longitud desc
    arr = sorted(toks.items(), key=lambda kv: -len(kv[1]))
    return arr

def encode_frag(frag: bytes, arr) -> list:
    s = frag.decode("latin-1")
    out = []
    i = 0
    n = len(s)
    while i < n:
        hit = -1
        for tid, t in arr:
            if t and s.startswith(t, i):
                hit = tid
                break
        if hit < 0:
            out.append(s[i].encode("latin-1")[0])
            i += 1
        else:
            out.append(hit)
            i += len(arr_by_id[hit])
    return out

arr_by_id = {}

def main():
    global arr_by_id
    arr = load_vocab()
    arr_by_id = dict(arr)

    jobs = [
        (r"adaptive\corpus_hf.bin", r"adaptive\corpus_hf_ids.bin"),
        (r"adaptive\corpus_peng.bin", r"adaptive\corpus_peng_ids.bin"),
        (r"adaptive\corpus_st.bin", r"adaptive\corpus_st_ids.bin"),
        (r"adaptive\corpus_ft_small.bin", r"adaptive\corpus_ft_ids.bin"),
    ]
    for src, dst in jobs:
        data = open(src, "rb").read()
        out_ids = []
        frag = bytearray()
        for b in data:
            if b in (1, 2, 3):
                if frag:
                    out_ids.extend(encode_frag(bytes(frag), arr))
                    frag = bytearray()
                out_ids.append(b)
            else:
                frag.append(b)
        if frag:
            out_ids.extend(encode_frag(bytes(frag), arr))
        with open(dst, "wb") as f:
            f.write(struct.pack("<I", len(out_ids)))
            f.write(struct.pack(f"<{len(out_ids)}I", *out_ids))
        rl = sum(len(arr_by_id.get(v, chr(v))) for v in out_ids[:20000]) / min(20000, len(out_ids))
        print(f"{dst.split(chr(92))[-1]}: {len(out_ids)} ids de {len(data)} bytes "
              f"({len(data)/max(len(out_ids),1):.2f} chars/token)")

if __name__ == "__main__":
    main()



