import os
import streamlit as st

from rag_core import find_pdfs, load_and_chunk, build_vectorstore, HRBot

st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 780px; padding-top: 2rem;}
    .src {font-size: 0.82rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if st.secrets.get("LANGCHAIN_API_KEY", None):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"


@st.cache_resource(show_spinner="Loading HR documents and building the index...")
def get_bot(key):
    pdfs = find_pdfs(("./hr_pdfs",))
    if not pdfs:
        return None, []
    chunks = load_and_chunk(pdfs)
    return HRBot(build_vectorstore(chunks), api_key=key), pdfs


st.title("🏢 HR Policy Assistant")
st.caption("Ask questions about company HR policies. Answers come only from the official HR documents.")

if not api_key:
    st.error("Add GROQ_API_KEY in Streamlit secrets (app Settings → Secrets).")
    st.stop()

bot, pdfs = get_bot(api_key)
if bot is None:
    st.error("No PDFs found. Put the HR PDFs in the `hr_pdfs/` folder of the repo.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("RAG chatbot: FAISS + MMR retrieval + Groq LLM, with out-of-scope guardrails.")
    st.subheader("Documents")
    for p in pdfs:
        st.write("📄", os.path.basename(p))
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.subheader("Try asking")
    for q in ["How many days of annual leave do I get?", "What is the notice period for resignation?"]:
        if st.button(q):
            st.session_state.pending = q

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if sources:
        with st.expander("📚 Sources"):
            for s in sources:
                st.markdown(f"**{s['file']}** — page {s['page']}")
                st.markdown(f"<div class='src'>{s['snippet']}</div>", unsafe_allow_html=True)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        render_sources(m.get("sources"))

question = st.chat_input("Ask an HR question...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            answer, docs = bot.ask(question)
        st.write(answer)
        seen, sources = set(), []
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({"file": key[0], "page": key[1], "snippet": d.page_content[:220].replace("\n", " ") + "..."})
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
