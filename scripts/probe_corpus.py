"""Pre-flight check on a candidate corpus, before any golden-set work starts.

Run this on a directory of PDFs *before* writing a single question. It answers the
three things that silently waste days if you find them out later:

  1. Is there a text layer at all, or is the PDF scanned?
  2. Does the text layer have real word spacing? (A PDF can look perfect in a
     viewer and still extract as "destinazioneadusinonpotabili..." — every
     hand-copied gold quote would then fail validation for no visible reason.)
  3. Is there author-provided structure to split on, and how much of the corpus
     is tables that naive extraction will flatten?

Usage:
    python -m scripts.probe_corpus                 # probes data/raw
    python -m scripts.probe_corpus path/to/pdfs

Exits non-zero if any document fails a hard check, so it can gate an ingest.
"""

from collections import Counter
from pathlib import Path
from typing import List
import re
import sys

from langchain_community.document_loaders import PyPDFLoader
from ftfy import fix_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A page with fewer characters than this almost certainly has no text layer.
MIN_CHARS_PER_PAGE = 200
# Healthy Italian prose averages ~5-7 characters per whitespace-separated token.
# Above this, the extractor is dropping space glyphs and gluing words together.
MAX_AVG_WORD_LEN = 8
# Italian function words. If spacing is intact these appear constantly; if the
# extractor ate the spaces they appear almost never, which is the cleanest signal.
SPACING_CANARIES = ["della", "delle", "degli", "sono", "comma", "articolo", "presente", "che", "non"]
MIN_CANARIES_PER_PAGE = 10

# Structural boundaries worth splitting on. Checked separately because which one
# a corpus uses is not knowable in advance -- the PoliMi regolamenti didattici
# turned out to have zero "Art. N" and numbered sections instead.
HEADING_PATTERNS = {
    "Art. N":     re.compile(r"(?im)^\s*art\.\s*\d+"),
    "Articolo N": re.compile(r"(?im)^\s*articolo\s+\d+"),
    "N. Title":   re.compile(r"(?m)^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+[A-ZÀ-Ü][^\n]{3,80}$"),
}

# Cross-references to other norms. High density means rare literal tokens that
# dense embeddings blur and BM25 scores sharply -- i.e. hybrid retrieval has a
# hypothesis on this corpus rather than being decoration.
LEGAL_REF = re.compile(r"(?i)\b(?:d\.?\s?lgs|d\.?p\.?r|l\.?r|d\.?m|art\.|comma|allegato)\b")

NUM_TOKEN = re.compile(r"\b\d+[.,]\d+\b|\b\d{1,4}\b")


def load_pages(pdf_path: Path):
    
    """One Document per page, with encoding repaired the same way ingest.py does."""
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.page_content = fix_text(page.page_content)
    return pages


def spacing_report(full_text: str, n_pages: int) -> tuple:

    """Detect an extractor that dropped its space glyphs.

    Returns (avg_word_len, canaries_per_page, is_broken). Checked on two
    independent signals because either alone has false positives: a document
    of long compound terms inflates word length, and a very short document
    can be canary-poor by chance.
    """
    words = re.findall(r"\S+", full_text)
    avg_len = sum(len(w) for w in words) / max(len(words), 1)

    low = full_text.lower()
    canaries = sum(len(re.findall(rf"\b{c}\b", low)) for c in SPACING_CANARIES)
    per_page = canaries / max(n_pages, 1)

    broken = avg_len > MAX_AVG_WORD_LEN and per_page < MIN_CANARIES_PER_PAGE
    return avg_len, per_page, broken


def table_fraction(pages) -> tuple:

    """Fraction of pages, and of characters, that look like flattened tables.

    Heuristic: a page dominated by short lines that are dense in numbers. Naive
    PDF extraction turns a table into exactly that -- a column of loose values
    whose header row is hundreds of characters away.
    """
    tab_pages = tab_chars = 0
    total_chars = sum(len(p.page_content) for p in pages)

    for page in pages:
        lines = [ln.strip() for ln in page.page_content.splitlines() if ln.strip()]
        if not lines:
            continue
        short_ratio = sum(1 for ln in lines if len(ln) < 40) / len(lines)
        numbers = len(NUM_TOKEN.findall(page.page_content))
        if short_ratio > 0.6 and numbers > 40:
            tab_pages += 1
            tab_chars += len(page.page_content)

    return tab_pages, (100 * tab_chars / total_chars if total_chars else 0.0)


def probe(pdf_path: Path) -> dict:

    pages = load_pages(pdf_path)
    full = "\n".join(p.page_content for p in pages)
    n_pages = len(pages)
    chars = len(full)
    per_page = chars // max(n_pages, 1)

    avg_word, canaries, broken_spacing = spacing_report(full, n_pages)
    tab_pages, tab_pct = table_fraction(pages)
    headings = {name: len(rx.findall(full)) for name, rx in HEADING_PATTERNS.items()}

    problems = []
    if per_page < MIN_CHARS_PER_PAGE:
        problems.append("SCANNED (no usable text layer)")
    if broken_spacing:
        problems.append("BROKEN SPACING (gold quotes will not validate)")
    if max(headings.values()) == 0:
        problems.append("NO STRUCTURE (nothing to split on)")

    return {
        "name": pdf_path.stem,
        "pages": n_pages,
        "chars": chars,
        "chars_per_page": per_page,
        "avg_word_len": avg_word,
        "canaries_per_page": canaries,
        "headings": headings,
        "legal_refs": len(LEGAL_REF.findall(full)),
        "table_pages": tab_pages,
        "table_pct": tab_pct,
        "problems": problems,
    }


def print_report(results: List[dict]) -> None:

    print(f"\n{'document':<34} {'pages':>6} {'chars':>9} {'ch/pg':>7} "
          f"{'wordlen':>8} {'refs':>6} {'%table':>7}  structure")
    print("-" * 110)
    for r in results:
        best = max(r["headings"], key=r["headings"].get)
        struct = f"{best} x{r['headings'][best]}" if r["headings"][best] else "none"
        print(f"{r['name'][:34]:<34} {r['pages']:>6} {r['chars']:>9} {r['chars_per_page']:>7} "
              f"{r['avg_word_len']:>8.1f} {r['legal_refs']:>6} {r['table_pct']:>6.0f}%  {struct}")

    print(f"\n{'-' * 110}")
    total_pages = sum(r["pages"] for r in results)
    total_chars = sum(r["chars"] for r in results)
    print(f"CORPUS: {len(results)} documents, {total_pages} pages, {total_chars:,} characters")
    # ~4 chars/token in Italian; text-embedding-3-small is $0.02 per 1M tokens.
    print(f"Rough embedding cost at 1000-char chunks: ${(total_chars / 4) * 0.02 / 1_000_000:.3f}")

    failed = [r for r in results if r["problems"]]
    if failed:
        print(f"\n{len(failed)} document(s) FAILED a hard check:")
        for r in failed:
            for p in r["problems"]:
                print(f"  {r['name']}: {p}")
            if "BROKEN SPACING (gold quotes will not validate)" in r["problems"]:
                print(f"    avg word length {r['avg_word_len']:.1f} chars, "
                      f"only {r['canaries_per_page']:.0f} Italian stopwords/page")
    else:
        print("\nAll documents passed. Safe to ingest.")

    print("\nFull heading counts (which boundary to split on is a per-corpus fact):")
    for r in results:
        counts = ", ".join(f"{k}={v}" for k, v in r["headings"].items())
        print(f"  {r['name'][:40]:<40} {counts}")


def main() -> int:

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "raw"
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()

    if not target.is_dir():
        print(f"Not a directory: {target}")
        return 2

    # Top-level only, matching DirectoryLoader's default (recursive=False) in
    # ingest.py -- so a data/raw/_archive/ subfolder is invisible to both.
    pdfs = sorted(target.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {target}")
        return 2

    print(f"Probing {len(pdfs)} PDF(s) in {target}")
    results = []
    for pdf in pdfs:
        print(f"  reading {pdf.name} ...")
        results.append(probe(pdf))

    print_report(results)
    return 1 if any(r["problems"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
