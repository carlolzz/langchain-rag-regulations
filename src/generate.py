
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


PROMPT_TEMPLATE = """
Answer the question based only on the following context.
{context}

Answer the question based on the above context: {user_query}

Please provide a clear, helpful answer using only the information from these documents.
If you can't find the answer, say 'Could not find any relevant information'.
"""


# Build the prompt context by concatenating the retrieved chunks
def build_context(relevant_docs: List[Tuple[Document, float]]) -> str:
    return "\n\n=====\n\n".join(doc.page_content for doc, _score in relevant_docs)


# History aware RAG -> query reformulation
def generate_answer(relevant_docs: List[Tuple[Document, float]], user_query: str, model_name: str = "gpt-4o") -> str:
    """From a given user question, generate answer with a given LLM and relevant context."""

    model = ChatOpenAI(model=model_name)

    context = build_context(relevant_docs)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, user_query=user_query)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt),
    ]

    # Invoke and call the LLM
    response_raw = model.invoke(messages)
    # Extracting the actual response without all the metadata, from the AI Message
    return response_raw.content
