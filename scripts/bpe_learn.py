"""entrena BPE vocab 2048 sobre los corpora existentes -> adaptive/vocab.bpe
formato: linea por merge "A B" (A,B
 son strings concatenables); ids: 0-255 bytes, 256+ merges
"""
import collections

def load_text():
    parts = []
    for fn in [r"C:\tinyllama\adaptive\corpus_hf.bin",
               r"C:\tinyllama\adaptive\corpus_peng.bin",
               r"C:\tinyllama\adaptive\corpus_ft_small.bin"]:
        parts.append(open(fn, "rb").read())
    data = b"".join(parts)
    # submuestra para velocidad: 60MB basta para un vocab 2048 solido
    if len(data) > 64_000_000:
        data = data[:64_000_000]
    return data

def get_words(data):
    """palabras=runs de letras (con acentos); resto separador. lower no (peticion)"""
    words = []
    cur = bytearray()
    for b in data:
        c = chr(b)
        if c.isalpha() or (0xC0 <= b <= 0xFF):
            cur.append(b)
        else:
            if cur:
                words.append(bytes(cur)); cur = bytearray()
    if cur: words.append(bytes(cur))
    return words

def train_bpe(data, n_merges=1792):
    word_counts = collections.Counter(get_words(data))
    print("palabras unicas:", len(word_counts))
    # each word = tuple of byte-ids
    vocab = {(b,): c for b in range(256) for c in ()}  # placeholder
    segs = {}   # word_bytes -> list of int ids (current segmentation)
    for w, c in word_counts.items():
        segs[w] = list(w)
    merges = []
    # id de cada merge = 256+idx directamente (no 100000+step)
    for step in range(n_merges):
        pairs = collections.Counter()
        for w, seg in segs.items():
            c = word_counts[w]
            for i in range(len(seg) - 1):
                pairs[(seg[i], seg[i+1])] += c
        if not pairs:
            break
        best, cnt = pairs.most_common(1)[0]
        if cnt < 2:
            print("pare de merges a", step, "(freq <2)")
            break
        merges.append(best)
        a, b = best
        nid = 256 + step
        for w, seg in segs.items():
            i = 0; ns = []
            while i < len(seg):
                if i < len(seg) - 1 and seg[i] == a and seg[i+1] == b:
                    ns.append(nid)
                    i += 2
                else:
                    ns.append(seg[i]); i += 1
            segs[w] = ns
        if step % 200 == 0:
            print("merge", step, "ok, mejor par", best, "cnt", cnt)
    return merges

def save_vocab(merges, path):
    # vocab.bpe: tabla. Linea: id  strA|strB
    with open(path, "w", encoding="utf-8") as f:
        f.write("# bpe vocab 2048: id  str_base  str_add  (merge)\n")
        for idx, (a, b) in enumerate(merges):
            sa = tok_str(a, merges)
            sb = tok_str(b, merges)
            f.write(f"{256+idx}\t{sa}\t{sb}\n")
    print("vocab guardado:", path)

def tok_str(tid, merges):
    # iterativo con mapa id->texto para evitar recursion sobre ids fantasma
    id2str = {i: (chr(i) if 32 <= i < 127 else bytes([i]).decode('latin-1')) for i in range(256)}
    for idx, (a, b) in enumerate(merges):
        id2str[256 + idx] = id2str.get(a, '?') + id2str.get(b, '?')
    return id2str.get(tid, '?')

if __name__ == "__main__":
    print("cargando corpora...")
    data = load_text()
    print("bytes:", len(data))
    print("entrenando 1792 merges (vocab 2048)...")
    merges = train_bpe(data)
    save_vocab(merges, r"C:\tinyllama\adaptive\vocab.bpe")