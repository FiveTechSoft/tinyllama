"""extrae texto de PDFs didacticos (mml-book + rohanmistry231) -> corpus_math.bin
el corpus matematico aporta tecnicismo estructurado (formulas en prosa, definiciones)
al vocabulario del modelo adaptativo"""
import os, sys
from pypdf import PdfReader

SOURCES = [
    r"C:\Users\Anto\Downloads\mml-book.pdf",
    # los del repo se descargan al directorio si existen
]

def extract_pdf(path):
    r = PdfReader(path)
    chunks = []
    for pg in r.pages:
        t = pg.extract_text() or ""
        # filtra paginas casi vacias
        if len(t.strip()) > 40:
            chunks.append(t)
    return chunks

def main(outs_dir=r"C:\tinyllama\adaptive"):
    out_path = os.path.join(outs_dir, "corpus_math.txt")
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pdf in SOURCES:
            if not os.path.exists(pdf):
                print("skip (no existe):", pdf)
                continue
            print("extrayendo:", os.path.basename(pdf))
            for chunk in extract_pdf(pdf):
                # registro formato corpus: control 1 + texto + control 3
                f.write("\x01" + " ".join(chunk.split()) + "\x03\n")
            n = sum(len(extract_pdf(pdf)) for _ in ())
            print("  pg extraidas:", len(chunk))
            total += len(chunk)
    print("total chars texto:", total)
    print("corpus_math.txt listo")

if __name__ == "__main__":
    main()