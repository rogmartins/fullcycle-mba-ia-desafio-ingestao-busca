import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("OPENAI_API_KEY", "DATABASE_URL","PGVECTOR_COLLECTION", "PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")


PDF_PATH = os.environ["PDF_PATH"]

def ingest_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(documents)
    if not splits:
        raise RuntimeError("No document splits were created from the PDF.")
    print(f"Created {len(splits)} document splits from the PDF.")
    enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in splits
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.environ["PGVECTOR_COLLECTION"],
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )
    store.add_documents(documents=enriched, ids=ids)
    


if __name__ == "__main__":
    ingest_pdf()