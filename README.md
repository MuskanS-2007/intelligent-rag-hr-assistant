# HR Policy RAG Chatbot

FAISS + MMR retrieval, Groq LLM (LCEL chain), out-of-scope guardrails, Streamlit UI.

## Files
- `hr_rag_notebook.ipynb` – upload to Kaggle (File > Import), add secrets, Run All -> `submission.csv`
- `rag_core.py` – pipeline (load, chunk, FAISS, LCEL chain, guardrail)
- `app.py` – Streamlit chat app with source citations
- `hr_pdfs/` – put the competition HR PDFs here (for deployment)

## Deploy
1. Push all files + `hr_pdfs/` to a GitHub repo.
2. share.streamlit.io -> New app -> select repo, main file `app.py`.
3. App Settings -> Secrets:
```
GROQ_API_KEY = "..."
LANGCHAIN_API_KEY = "..."
```
