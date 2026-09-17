
from langchain_core.documents import Document
from src.retrieval import rrf_fuse, tokenize

def test_tokenize_keeps_statute_numbers_whole():
    assert tokenize("DPR 380/2001, m 3,00") == ["dpr", "380/2001", "m", "3,00"]

def test_rrf_rewards_agreement():
    a, b, c = (Document(page_content=t, metadata={"chunk_id": t}) for t in "abc")
    fused = rrf_fuse([[(a, .9), (b, .8)], [(b, 12.0), (c, 5.0)]])
    assert [d.page_content for d, _ in fused] == ["b", "a", "c"]