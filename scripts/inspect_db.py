
# scripts/inspect_db.py
import chromadb

from src.config import CHROMA_PATH, SEP_WIDTH


def inspect(chroma_path: str = CHROMA_PATH) -> None:

    client = chromadb.PersistentClient(path=str(chroma_path))

    for collection in client.list_collections():
        print("=" * SEP_WIDTH)
        c = client.get_collection(collection.name)
        peek = c.peek(limit=1)
        print(f"\n{c.name}  —  {c.count()} chunks")
        # distance function, etc.
        print(f" - collection metadata: {c.metadata}")                
        if peek["ids"]: 
            print(f" - first id:    {peek['ids'][0]}")   
            # source, article, ...
            print(f" - chunk meta:  {peek['metadatas'][0]}")          
            print(f" - text:        {peek['documents'][0][:120]}...")


if __name__ == "__main__":
    inspect()
