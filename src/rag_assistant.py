"""Module 4b -- The retrieval-augmented maintenance assistant.

Loads the persisted ChromaDB store, retrieves the most relevant chunks for a
question and asks an LLM to answer using only those chunks.
"""

from __future__ import annotations

try:
    from src import config
    from src.build_rag import get_embeddings
except ImportError:
    import config
    from build_rag import get_embeddings


SYSTEM_PROMPT = """You are a maintenance assistant for a smart manufacturing plant.

Answer the engineer's question using ONLY the context below. The context comes
from the plant's own maintenance manuals and troubleshooting guides.

Rules:
- If the context does not contain the answer, say so plainly. Do not invent
  thresholds, part numbers or procedures.
- Be concise and practical. Prefer numbered steps for any procedure.
- Quote specific numbers (temperatures, speeds, thresholds) when the context
  gives them.

Context:
{context}

Question: {question}

Answer:"""


def knowledge_base_ready() -> bool:
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


def get_retriever(k: int = 4):
    from langchain_chroma import Chroma

    store = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name="maintenance_kb",
    )
    return store.as_retriever(search_kwargs={"k": k})


def get_llm():
    """Anthropic by default, Google Gemini if LLM_PROVIDER=google."""
    if config.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
        )

    from langchain_anthropic import ChatAnthropic

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
    return ChatAnthropic(
        model=config.ANTHROPIC_MODEL, temperature=0, max_tokens=1000,
        api_key=config.ANTHROPIC_API_KEY,
    )


def ask(question: str, k: int = 4) -> dict:
    """Returns {'answer': str, 'sources': [{'source', 'snippet'}]}."""
    from langchain_core.prompts import PromptTemplate

    docs = get_retriever(k).invoke(question)
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    chain = prompt | get_llm()
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": [
            {
                "source": d.metadata.get("source", "unknown"),
                "snippet": d.page_content[:300].replace("\n", " ") + "...",
            }
            for d in docs
        ],
    }


def search_only(question: str, k: int = 4) -> list[dict]:
    """Retrieval without an LLM call -- lets the app still be demonstrated
    when no API key is configured."""
    docs = get_retriever(k).invoke(question)
    return [
        {"source": d.metadata.get("source", "unknown"), "snippet": d.page_content}
        for d in docs
    ]
