
from src.retrieval import retrieve, get_sources
from src.generate import generate_answer
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import List, Optional, Tuple
from src.config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL, get_collection_name

SEP_WIDTH = 100
CHAT_HISTORY = []
MAX_HISTORY_TURNS = 4
NO_CONTEXT_MSG = "Could not find any relevant information in the corpus."
REWRITE_SYSTEM_PROMPT = """Given the chat history, rewrite the user's new question so that it can be understood on its own, without the history. 
Resolve pronouns and implicit references into explicit terms. Do not answer the question. 
If the question is already self-contained, return it unchanged. Return only the rewritten question, with no preamble.
"""


def rewrite_query(query: str, history: List[BaseMessage], model_name: str) -> str:

    if not history:
        return query

    model = ChatOpenAI(model=model_name)
    messages = (
        [SystemMessage(content=REWRITE_SYSTEM_PROMPT)]
        + history
        + [HumanMessage(content=f"New question: {query}")]
    )

    return model.invoke(messages).content.strip()


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


def ask(query: str, emb_model_name: str, llm_model: str, history: Optional[List[BaseMessage]] = None, history_aware: bool=True) -> Tuple[str, set]:

    # Not a mutable default argument, the list would be shared by every call
    history = history if history is not None else []
    search_query = rewrite_query(query, history, llm_model) if history_aware else query

    # Get relevant documents
    # Collection name not hardcoded
    relevant_docs = retrieve(
        chroma_path=CHROMA_PATH, 
        query=search_query, 
        collection_name=get_collection_name(),
        embedding_model=emb_model_name
    ) 

    if not relevant_docs:
        answer, sources = NO_CONTEXT_MSG, set()
    else:
        answer = generate_answer(relevant_docs, query, llm_model)
        sources = get_sources(relevant_docs)

    # Add context to the history
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=answer))

    # Two messages per turn
    max_messages = 2 * MAX_HISTORY_TURNS
    # Example, length of history is 10, we delete up to 10 - 8 = first 2 messages
    if len(history) > max_messages:
        del history[: len(history) - max_messages]

    return answer, sources


def start_chat_cli():

    print("Ask me questions! Type '!quit' or '!q' to exit.")

    history: List[BaseMessage] = []

    while True:
        question = input("\nYour question:").strip()

        if not question:
            continue
        if question.lower() in {"!quit", "!q"}:
            break
        answer, sources = ask(query=question, model_name=EMBEDDING_MODEL, llm_model=LLM_MODEL, history=history, history_aware=True)
        print_response(answer, sources)


if __name__ == "__main__":
    start_chat_cli()