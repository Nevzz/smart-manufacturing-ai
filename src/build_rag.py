"""Module 4a -- Build the vector knowledge base.

Reads every document in data/docs/, splits it into overlapping chunks, embeds
them and stores them in a local ChromaDB collection.

Run:  python src/build_rag.py
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from src import config
except ImportError:
    import config


class FastEmbedEmbeddings(Embeddings):
    """Thin LangChain wrapper around FastEmbed.

    FastEmbed runs on ONNX Runtime, so it needs no PyTorch -- which matters
    here because TensorFlow is already installed for the vision module.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()


def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings()


def load_documents() -> list[Document]:
    """Loads .md / .txt directly and .pdf via pypdf. One Document per file."""
    files = sorted(
        p for p in config.DATA_DOCS.iterdir()
        if p.suffix.lower() in {".md", ".txt", ".pdf"}
    )
    if not files:
        raise SystemExit(f"No documents found in {config.DATA_DOCS}")

    docs: list[Document] = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            text = "\n\n".join(
                (page.extract_text() or "") for page in PdfReader(str(path)).pages
            )
        else:
            text = path.read_text(encoding="utf-8")

        docs.append(Document(page_content=text, metadata={"source": path.name}))
        print(f"  {path.name}  ({len(text):,} chars)")

    return docs


def main() -> None:
    print("=" * 60)
    print("MODULE 4 -- Building the maintenance knowledge base")
    print("=" * 60)

    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"\nReading documents from {config.DATA_DOCS.relative_to(config.ROOT)}")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"\nSplit into {len(chunks)} chunks (800 chars, 120 overlap)")

    print("Embedding and writing to ChromaDB (first run downloads the model)...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_name="maintenance_kb",
    )
    print(f"Stored {store._collection.count()} vectors -> "
          f"{config.CHROMA_DIR.relative_to(config.ROOT)}")
    print("\nDone. Next: streamlit run app/main.py\n")


if __name__ == "__main__":
    main()
