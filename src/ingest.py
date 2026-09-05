
from collections import Counter, defaultdict
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from ftfy import fix_text
from pathlib import Path
from typing import List
import hashlib
import re

from src.config import EMBEDDING_MODEL, EMBEDDING_ABBR, CHUNKING_STRATEGY, CHUNKING_CFG, F_EXT, get_collection_name


# Load environment variables (OPENAI_API_KEY)
load_dotenv()


# parent -> parent, only 1 .parent gives src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
KEY_PREFIX = 60

# Which loader (and its kwargs) handles each supported file extension
LOADER_CFG = {
    "pdf" : (PyPDFLoader, {}),
    "txt" : (TextLoader, {"encoding": "utf-8"}),
    "md"  : (TextLoader, {"encoding": "utf-8"}),
}

# Which splitter suits the structure of each supported file extension.
SPLITTER_CFG = {
    "pdf" : (RecursiveCharacterTextSplitter, "rcts"),
    "txt" : (RecursiveCharacterTextSplitter, "rcts"),
    "md"  : (MarkdownTextSplitter, "mdts")
}


# Load all documents in the data/raw directory
def load_documents(data_path="data/raw", ext:str="pdf") -> List[Document]:

    docs_path = PROJECT_ROOT / data_path
    docs_path = docs_path.resolve()

    if not Path.exists(docs_path):
        raise FileNotFoundError(f"Error, directory {docs_path} not found.")

    if ext not in LOADER_CFG:
        supported = ", ".join(sorted(LOADER_CFG.keys()))
        raise ValueError(f"Error, unsupported extension .{ext}. Supported: {supported}.")

    loader_cls, loader_kwargs = LOADER_CFG[ext]
    # e.g. pdf - PyPDFLoader

    loader = DirectoryLoader(
        path=docs_path,
        glob=f"*.{ext}",
        loader_cls=loader_cls,
        loader_kwargs=loader_kwargs,
    )

    docs = loader.load()

    if len(docs) == 0:
        raise FileNotFoundError(f"Error, no .{ext} files found in {docs_path}.")

    # Store the source as a path relative to the project root, e.g. "data/raw/foo.pdf".
    # The absolute path would change if the project folder is ever moved or renamed
    # This would change every chunk ID and silently duplicate the whole corpus.
    for doc in docs:
        source = Path(doc.metadata["source"]).resolve()
        doc.metadata["source"] = source.relative_to(PROJECT_ROOT).as_posix()

    n = 3
    # Show first n documents
    for idx, doc in enumerate(docs[:n]):
        print(f"\nDocument {idx + 1}")
        print(f" - Source: {doc.metadata['source']}")
        print(f" - Content length: {(len(doc.page_content))} characters")
        print(f" - Content preview: {doc.page_content[:100]}...")
        print(f" - Metadata: {doc.metadata}")

    return docs


# whitespaces -> " " -> digits -> #
def _key(line: str) -> str:
    norm: str = re.sub(r"\d+", "#", re.sub(r"\s+", " ", line)).strip().lower()
    return norm[:KEY_PREFIX]


def strip_page_furniture(docs, min_page_fraction: float = 0.8, min_len: int = 12) -> List[Document]:
    """Drop lines that repeat on nearly every page of the same document"""

    # Pairs [docs metadata source (file and page), document itself]
    by_source = defaultdict(list)
    for d in docs:
        by_source[d.metadata["source"]].append(d)

    for _source, pages in by_source.items():
        counts = Counter()
        for p in pages:
            # Take every line in the page content, apply _key regex function
            # Update the count of said line if the lenght is longer than min_len
            counts.update({_key(line) for line in p.page_content.splitlines() if len(line.strip()) >= min_len})
        # line is boilerplate if it appears min_page_fraction * num_of_pages time
        boilerplate = {line for line, cnt in counts.items() if cnt >= min_page_fraction * len(pages)}
        for p in pages:
            p.page_content = "\n".join(
                line for line in p.page_content.splitlines() if _key(line) not in boilerplate
            )
    return docs


# Repair mojibake left behind by PDF text extraction, in place.
# e.g. "societÃ  di ingegneria" -> "società di ingegneria"
def clean_documents(docs: List[Document]) -> List[Document]:

    n_fixed = 0
    for doc in docs:
        fixed = fix_text(doc.page_content)
        if fixed != doc.page_content:
            doc.page_content = fixed
            n_fixed += 1

    print(f"\nRepaired text encoding in {n_fixed}/{len(docs)} documents.")
    return docs


def split_documents(docs: List[Document], chunking_strategy, ext:str="pdf") -> List[Document]:

    cfg = CHUNKING_CFG.get(chunking_strategy)
    if not cfg:
        raise ValueError(f"Unknown chunking strategy '{chunking_strategy}'. Options: {', '.join(CHUNKING_CFG)}")
    
    chunk_size, chunk_overlap = cfg["chunk_size"], cfg["chunk_overlap"]

    if ext not in SPLITTER_CFG:
        supported = ", ".join(sorted(SPLITTER_CFG))
        raise ValueError(f"Error, unsupported extension .{ext}. Supported: {supported}.")

    # "pdf" : (RecursiveCharacterTextSplitter, "RCS"),
    splitter_cls, _abbr = SPLITTER_CFG[ext]

    print(f"Splitting .{ext} documents into chunks with {splitter_cls.__name__}...")

    text_splitter = splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs)

    n = 5
    if chunks:
        for idx, chunk in enumerate(chunks[:n]):
            print(f"=" * 100)
            print(f"\n===== Chunk {idx + 1} =====")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}")
            print(f"=" * 100)

    if len(chunks) > n:
        print(f"\n{len(chunks) - n} more chunks remaining.")

    return chunks


# Give every chunk a stable, deterministic ID: hash of source + index within that source + content. 
# Re-running ingest on unchanged files produces the same IDs
# Chroma upserts instead of appending, and the store stays idempotent.
def build_chunk_ids(chunks: List[Document]) -> List[str]:

    ids = []
    # How many chunks we have already seen for each source document
    chunks_per_source = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = chunks_per_source.get(source, 0)
        chunks_per_source[source] = idx + 1

        raw_id = f"{source}|{idx}|{chunk.page_content}"
        chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

        # Keep the ID on the chunk too, so the golden set can point at specific chunks
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = idx
        ids.append(chunk_id)

    return ids


def create_vector_store(chunks: List[Document], chunking_strategy: str, ext: str, persist_dir="db/chroma_db", replace_db:bool=True):

    print(f"Creating embeddings and storing them in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_path = PROJECT_ROOT / persist_dir
    persist_path = persist_path.resolve()

    collection_name = get_collection_name(chunking_strategy, ext, EMBEDDING_MODEL)

    if replace_db:
        try:
            Chroma(
                persist_directory=str(persist_path),
                embedding_function=embedding_model,
                collection_name=collection_name,
            ).delete_collection()
        except Exception:
            pass

    chunk_ids = build_chunk_ids(chunks)

    print(f"Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=chunk_ids,
        persist_directory=persist_path,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    stored = vector_store._collection.count()
    print(f"Successfully created the vector store and saved it to {persist_path}.")
    print(f"{len(chunks)} chunks ingested, {stored} total in the collection.")
    return vector_store


def detect_extension(data_path: Path) -> str:

    # get extensions of all files in data/raw
    exts = {p.suffix.lstrip(".").lower() for p in data_path.iterdir() if p.is_file()}
    # set intersection with existing allowed extensions
    exts &= LOADER_CFG.keys()
    if not exts:
        raise FileNotFoundError(f"No supported files in {data_path}. Supported: {sorted(LOADER_CFG)}")
    if len(exts) > 1:
        raise ValueError(f"Mixed extensions in {data_path}: {sorted(exts)}. Ingest one type per run.")

    return exts.pop()


if __name__ == "__main__":

    # Loading the files, cleaning them, splitting them in chunks, and saving them to a vector db
    # Not hardcoding the extension, let the function get it
    ext = detect_extension(DATA_RAW_DIR)

    if ext != F_EXT:
        raise ValueError(
            f"config.EXT is {F_EXT} but {DATA_RAW_DIR} contains .{ext}"
            f"The collection name derives from this, update one or the other."
        )

    raw_docs = load_documents(ext=ext)
    cleaned_docs = strip_page_furniture(clean_documents(raw_docs))
    chunks = split_documents(cleaned_docs, CHUNKING_STRATEGY, ext)
    vector_store = create_vector_store(chunks, CHUNKING_STRATEGY, ext, persist_dir="db/chroma_db", replace_db=True)