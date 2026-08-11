
from src.retrieval import retrieve, get_sources
from src.generate import generate_answer


CHROMA_PATH = "db/chroma_db"
SEP_WIDTH = 100


# Print a nicely formatted output to the user
def print_response(response_text: str, sources: set) -> None:

    sources_formatted = "\n".join([f" - {source}" for source in sources])

    print("\n" + "=" * SEP_WIDTH)
    print("LLM Response")
    print("=" * SEP_WIDTH)
    print(f"\n{response_text}\n")
    print("-" * SEP_WIDTH)
    print(f"Sources:\n{sources_formatted}")
    print("=" * SEP_WIDTH + "\n")


def ask():
    
    usr_queries = [
        "Quali sono i requisiti per iscriversi al corso di laurea magistrale computer science and engineering?",
        "Che media devo avere per essere ammesso al corso di laurea magistrale computer science and engineering?"
    ]

    for q in usr_queries:
        relevant_docs = retrieve(chroma_path=CHROMA_PATH, query=q)
        if not relevant_docs:
            continue

        response_text = generate_answer(relevant_docs, q)
        print_response(response_text, get_sources(relevant_docs))


if __name__ == "__main__":
    ask()