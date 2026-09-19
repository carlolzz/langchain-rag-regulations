
import re
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


# A heading is "Art. 12", "Art.12", "ARTICOLO 12", "Articolo 12 bis" - or "ALLEGATO A" - at the START
# of a line. Without the ALLEGATO branch, every annex is labelled as the last article.
ARTICLE_RE = re.compile(
    r"^[ \t]*(?:(?:art\.|articolo)[ \t]*(\d+(?:[ \t]*(?:bis|ter|quater))?)\b|(allegato[ \t]+['\"‘’]?[a-z0-9]+))",
    re.IGNORECASE | re.MULTILINE,
)
# Table-of-contents lines: "Art.19 Richiesta ........ 20" or "Art. 33 - Cortili 29"
TOC_RE = re.compile(r"\.{4,}|\s\d{1,3}\s*$")

# "art. 5", "n. 380", "m. 2,10" - a dot followed by a space that does not end a sentence
ABBR_RE = re.compile(
    r"\b(artt?|nn?|co|comma|lett|ecc|cfr|pag|es|par|cap|all|tab|fig|vol|mq|mc|ml|cm|mm|km|m|h|min|max|sig|dott|ing|arch|geom)\.(?=\s)",
    re.IGNORECASE,
)
# Dotted acronyms: "D.Lgs.", "D.P.R.", "s.m.i.", "L.R."
ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]{1,4}\.){2,}")
DOT = "<DOT>"


def join_pages(docs: List[Document]) -> List[Document]:
    """One document per source. Article span pages, so splitting by page wouldn't work."""

    # result = [torino_doc, firenze_doc, ...]
    # by_source = {"torino.pdf" -> [doc_1, doc_2 ...]}
    by_source = defaultdict(list)
    for doc in docs:
        by_source[doc.metadata["source"]].append(doc)

    result: List[Document] = []
    # source: key, pages: value
    for source, pages in by_source.items():
        doc = Document(
            page_content="\n".join(p.page_content for p in sorted(pages, key=lambda p: p.metadata.get("page", 0))),
            metadata={"source": source} 
        )
        result.append(doc)

    return result


def _line_at(text: str, start: int) -> str:
    # The line that starts at `start`, without its newline, so TOC_RE checks only the heading's own line
    end = text.find("\n", start)
    return text[start:] if end == -1 else text[start:end]


def split_documents_by_article(docs: List[Document], max_chars: int = 2000, overlap: int = 0) -> List[Document]:
    """Split by article headings, tag each piece with its article, sub-split anything too long"""

    # Built once, outside the loop, and reused for every article. It only cuts text longer than chunk_size.
    sub_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    chunks: List[Document] = []

    for doc in join_pages(docs):

        text, source = doc.page_content, doc.metadata["source"]

        # First ARTICLE_RE.finditer finds every heading, TOC entries included (one Match per heading, `^` works on every line thanks to re.MULTILINE).
        # Then TOC_RE checks each heading's own line: dot leaders or a page number at the end means it's a TOC entry, so it's dropped.
        # h.start() is where the heading starts in the text, e.g. character 30
        # TOC_RE.search() gives a Match (truthy) or None (falsy), so `not ...` keeps only the real headings
        headings =[h for h in ARTICLE_RE.finditer(text) if not TOC_RE.search(_line_at(text, h.start()))]

        if not headings:
            raise ValueError(f"No article headings found in {source}.")

        # bounds is a list of (offset, label) pairs, one per heading
        # offset = the character where the heading starts, counted from the start of the document
        # label = group 1 if found (article number), otherwise group 2 (allegato), since the missing group is None
        # Then it's cleaned up: whitespace and quotes become one space, lowercased, e.g. "Art. 12  bis" -> "12 bis", "ALLEGATO 'A'" -> "allegato a"
        bounds = [(h.start(), re.sub(r"[\s'\"‘’]+", " ", h.group(1) or h.group(2)).strip().lower()) for h in headings]
        # bounds[0][0] is the offset of the first heading (first pair, first element)
        # If it isn't 0 there's text before the first article, so we add a "preambolo" boundary at the front
        if bounds[0][0] > 0:
            bounds.insert(0, (0, "preambolo"))

        # zip gives the current heading and the next one at each step
        # The last heading has no next one, so (len(text), None) makes it run to the end of the text
        # article = the heading label, start and end = character numbers, e.g. from character 30 to character 100
        for (start, article), (end, _next) in zip(bounds, bounds[1:] + [(len(text), None)]):
            # So the body is the text from one heading's start up to the next heading, heading line included
            # text is the doc page content, text[30:100] = characters 30 to 99
            body = text[start:end].strip()
            # split_text() takes a str and returns a list of str (split_documents() works on Documents instead)
            # If the article is shorter than max_chars it comes back as a list with one element, so the loop runs once
            for piece in sub_splitter.split_text(body):
                chunks.append(Document(page_content=piece, metadata={"source": source, "article": article}))

    return chunks


# Semantic chunking splits where the topic changes, not at a fixed length. It costs one embedding per sentence.
# First it splits the text into sentences and embeds each one together with its neighbours (buffer_size=1)
# Then it gets the distance between each sentence and the next one, distance = 1 - cosine similarity
# So it cuts wherever the distance is above the percentile, e.g. percentile=95 means only the top 5% of gaps become cuts
# Higher percentile = fewer and longer chunks, and there's no max size
def split_documents_semantics(docs: List[Document], percentile: float, embedding_model: str) -> List[Document]:

    # The chunker only holds the embeddings object here, nothing gets embedded (and no API call) until split_documents()
    # threshold_type is the statistic used for the cut ("percentile", "standard_deviation", "interquartile", "gradient")
    # threshold_amount is its value, for percentile it goes from 0 to 100, e.g. 95, not 0.95
    chunker = SemanticChunker(
        OpenAIEmbeddings(model=embedding_model),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile
    )

    joined = join_pages(docs)
    # doc.page_content = ... changes the Document inside joined directly, since Documents are mutable
    # doc = ... instead would only change the loop variable, and joined would stay the same
    for doc in joined:
        doc.page_content = protect_abbreviations(doc.page_content)

    # split_documents() takes a list of Documents and gives back new ones, each with a copy of its parent's metadata
    # So every chunk keeps its "source" without doing anything extra
    chunks = chunker.split_documents(joined)
    # Same thing as above but on the chunks, this puts the real dots back
    for chunk in chunks:
        chunk.page_content = restore_abbreviations(chunk.page_content)

    return chunks


# SemanticChunker splits sentences after every . ? ! followed by a space, so "art. 5" or "D.P.R. 380" would end a sentence
# So before chunking those dots become DOT. Acronyms go first, so "D.P.R." is replaced all at once
def protect_abbreviations(text: str) -> str:
    text = ACRONYM_RE.sub(lambda m: m.group(0).replace(".", DOT), text)
    return ABBR_RE.sub(lambda m: m.group(1) + DOT, text)


# After chunking DOT goes back to ".", so the stored text is the same as the PDF and the gold quotes still match
def restore_abbreviations(text: str) -> str:
    return text.replace(DOT, ".")