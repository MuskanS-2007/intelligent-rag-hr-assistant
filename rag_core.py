"""Shared RAG pipeline for the HR policy chatbot (used by the Kaggle notebook and the Streamlit app)."""
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

REFUSAL = (
    "I'm sorry, but I can only answer questions about our company's HR policies "
    "based on the HR documents provided. This question is outside the scope of those "
    "documents, so I can't answer it. Please contact the HR department for further help."
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
# Minimum relevance of the best retrieved chunk (0..1). Below this -> treated as out of scope.
MIN_RELEVANCE = 0.15

PROMPT = ChatPromptTemplate.from_template(
    """You are an HR policy assistant for company employees.
Answer the question using ONLY the context below, which comes from the company's HR documents.

Rules:
1. Be accurate and specific: include exact numbers, durations, amounts, eligibility rules and conditions as written in the context.
2. Answer in 1-4 clear sentences of plain text (no markdown, no bullet symbols).
3. Do NOT use outside knowledge or guess.
4. If the question is not about the HR documents, or the context does not contain the answer, reply with EXACTLY this text and nothing else:
{refusal}

Context:
{context}

Question: {question}

Answer:"""
)


def find_pdfs(root_dirs=("/kaggle/input", "./data", ".")):
    pdfs = []
    for root in root_dirs:
        pdfs += glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True)
    return sorted(set(pdfs))


def load_and_chunk(pdf_paths, chunk_size=800, chunk_overlap=150):
    docs = []
    for p in pdf_paths:
        pages = PyPDFLoader(p).load()
        for pg in pages:
            pg.metadata["source"] = os.path.basename(p)
            pg.metadata["page"] = int(pg.metadata.get("page", 0)) + 1
        docs.extend(pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_documents(chunks, emb)


def _format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source')} p.{d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )


class HRBot:
    def __init__(self, vectorstore, api_key=None, k=5, fetch_k=20):
        self.vs = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.6}
        )
        self.llm = ChatGroq(
            model=LLM_MODEL, temperature=0, api_key=api_key or os.environ.get("GROQ_API_KEY")
        )
        # LCEL chain
        self.chain = (
            {
                "context": RunnableLambda(lambda x: _format_docs(x["docs"])),
                "question": RunnableLambda(lambda x: x["question"]),
                "refusal": RunnableLambda(lambda x: REFUSAL),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    def _in_scope(self, question):
        res = self.vs.similarity_search_with_relevance_scores(question, k=1)
        return bool(res) and res[0][1] >= MIN_RELEVANCE

    def ask(self, question):
        """Returns (answer, source_documents)."""
        if not self._in_scope(question):
            return REFUSAL, []
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke({"question": question, "docs": docs}).strip()
        if answer.startswith("I'm sorry, but I can only answer"):
            return REFUSAL, []
        return answer, docs
