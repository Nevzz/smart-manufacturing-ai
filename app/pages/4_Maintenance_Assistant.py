import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src import config, rag_assistant

st.set_page_config(page_title="Maintenance Assistant", page_icon="💬", layout="wide")
st.title("💬 AI Maintenance Assistant")
st.caption("Retrieval-augmented chatbot over the plant's maintenance documents")

if not rag_assistant.knowledge_base_ready():
    st.warning("Knowledge base not built yet. Run `python src/build_rag.py` first.")
    st.stop()

has_key = bool(config.ANTHROPIC_API_KEY or config.GOOGLE_API_KEY)

with st.sidebar:
    st.subheader("Knowledge base")
    for p in sorted(config.DATA_DOCS.glob("*")):
        if p.suffix.lower() in {".md", ".txt", ".pdf"}:
            st.caption(f"• {p.name}")
    k = st.slider("Chunks to retrieve", 2, 8, 4)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if not has_key:
    st.info(
        "No LLM API key found in `.env`, so the assistant is running in "
        "**retrieval-only mode** — it will show the relevant manual passages "
        "without writing an answer. Add `ANTHROPIC_API_KEY` to enable full answers."
    )

EXAMPLES = [
    "Tool wear is at 210 minutes. What should I do?",
    "Temperature difference has dropped to 8 K. What is causing this?",
    "The model flagged a 0.7 failure probability. How urgent is that?",
    "What causes rolled-in scale and how do I fix it?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", width="stretch"):
            st.session_state.pending = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}**")
                    st.caption(s["snippet"])

question = st.chat_input("Ask about a failure mode, a defect or a procedure...")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the manuals..."):
            try:
                if has_key:
                    result = rag_assistant.ask(question, k=k)
                    answer, sources = result["answer"], result["sources"]
                else:
                    hits = rag_assistant.search_only(question, k=k)
                    answer = "**Retrieval-only mode — most relevant passages:**\n\n" + \
                        "\n\n".join(f"*From {h['source']}*\n\n{h['snippet']}" for h in hits)
                    sources = []
            except Exception as exc:
                answer = f"Something went wrong: `{exc}`"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for s in sources:
                    st.markdown(f"**{s['source']}**")
                    st.caption(s["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
