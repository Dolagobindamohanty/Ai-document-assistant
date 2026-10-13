# 📚 AI Document Assistant

An evidence-grounded, production-grade Retrieval-Augmented Generation (RAG) assistant for querying, summarizing, and studying unstructured documents (PDFs and text notes) from laptops and mobile browsers.

Powered by **Streamlit**, **LangChain**, **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**, **FAISS Vector Indexing**, and **Google Gemini (`google-genai`)** with automated multi-tier model failover.

---

## 🌟 Key Capabilities

1. **Evidence-Grounded Q&A (`Ask questions`):**
   - Retrieves diverse candidate chunks using **Maximal Marginal Relevance (MMR)** (`k=5`, `fetch_k=10`, `lambda=0.7`).
   - Applies **relevance distance gating** (`max_distance=1.90`) to filter out-of-domain queries and reject ungrounded hallucinations with explicit refusals: *"I don't know based on the provided documents."*
   - Returns verifiable source citations in the standard format: `[filename.pdf, p. 12]`.

2. **Full-Document Hierarchical Summarization (`Summarize documents`):**
   - Does **not** take shortcuts using top-$k$ search.
   - Partitions **100% of extractable document pages** into context-bounded batches (12,000 characters).
   - Executes multi-stage **Map-Reduce** with live progress tracking and honest failure reporting (distinguishing `COMPLETE` vs `PARTIAL` coverage).

3. **Study Assets & Exam Prep (`Study tools`):**
   - **Study Notes:** Generates structured revision notes with headings, key definitions, and critical facts.
   - **Multiple-Choice Quizzes:** Formulates 5-question quizzes with schema validation (question, 4 options, valid 0-based answer index, and explanations).

4. **Multi-Tier Model Resilience:**
   - Primary model: `gemini-3.8-flash`.
   - Fallback cascade: `gemini-3.6-flash` $\rightarrow$ `gemini-3.5-flash-lite`.
   - Bounded exponential backoff with random jitter for transient 503 / 500 server errors.
   - Distinct, non-retryable handling for 429 quota exhaustion, 401/403 authentication, and 404 model errors.

5. **Security & Untrusted Content Protection:**
   - Document text is isolated inside `<CONTEXT>` blocks and strictly treated as untrusted data, neutralizing prompt injection attacks.
   - Automatic API key secret redaction across all error messages and logs.

---

## 🏗️ System Architecture

```text
Browser (Desktop / Mobile LAN)
        │
        ▼
Streamlit UI (app.py)
        │
        ├──► Document Service (src/services/document_service.py)
        │       └── Validation, SHA-256 Duplication Check, Page Extraction, Chunking
        │
        ├──► Index Service (src/services/index_service.py)
        │       └── Isolated In-Memory FAISS Vector Index (Zero legacy faiss_db mutation)
        │
        ├──► RAG Service (src/services/rag_service.py)
        │       └── MMR Diversity Retrieval, Distance Filtering, Context Budgeting
        │
        ├──► Summary Service (src/services/summary_service.py)
        │       └── 100% Page Coverage, Map-Reduce Synthesis, Coverage Manifest
        │
        ├──► Study Service (src/services/study_service.py)
        │       └── Study Notes Generation & Schema-Validated Multiple-Choice Quizzes
        │
        └──► Export Service (src/services/export_service.py)
                └── Markdown / Plain-text Downloads with Citations & Answer Keys
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.12, 3.13, or 3.14
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Environment Setup
Clone or navigate to the repository folder:
```bash
cd "RAG PROJECT"
```

Activate the virtual environment:
```powershell
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Edit `.env` and set your actual `GEMINI_API_KEY`:
```ini
GEMINI_API_KEY="AIzaSy..."
```

---

## 💻 Running the Application

### Local Desktop Access:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Mobile / Phone Access on Trusted Wi-Fi (LAN):
1. Bind Streamlit to the network interface:
```bash
python -m streamlit run app.py --server.address 0.0.0.0
```
2. Look for the **Network URL** in the terminal (e.g., `http://192.168.1.X:8501`).
3. Open this URL in the browser on your phone connected to the same Wi-Fi network.

---

## 🧪 Automated Testing

The project includes an automated test suite verifying all services, models, and error states without consuming live Gemini API credits:

```bash
# Run the complete test suite:
pytest -v

# Run individual test modules:
pytest tests/unit/test_llm.py -v              # LLM resilience, retries, 429 quota, auth
pytest tests/unit/test_document_service.py -v # PDF validation, extraction, chunking
pytest tests/unit/test_index_service.py -v    # FAISS isolation
pytest tests/unit/test_rag_service.py -v      # Grounded Q&A, abstention, citations
pytest tests/unit/test_summary_service.py -v  # Map-Reduce full document summarization
pytest tests/unit/test_study_service.py -v    # Study notes, quiz schemas, exports
pytest tests/unit/test_security.py -v         # Prompt injection defense
pytest tests/integration/test_app_smoke.py -v # Streamlit UI runtime initialization
```

---

## ⚙️ Configuration Reference (`src/config.py`)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | `""` | Required Google GenAI API credential |
| `GEMINI_PRIMARY_MODEL` | `gemini-3.8-flash` | Primary text generation model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.6-flash,gemini-3.5-flash-lite` | Ordered fallback models |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense vector embedding model |
| `CHUNK_SIZE` | `800` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `120` | Boundary overlap between adjacent chunks |
| `RETRIEVAL_K` | `5` | Top-$k$ results returned to context |
| `RETRIEVAL_FETCH_K` | `10` | MMR candidate pool size |
| `MMR_LAMBDA` | `0.7` | Relevance vs. diversity trade-off |
| `MAX_DISTANCE` | `1.90` | L2 distance ceiling for relevance gating |
| `MAX_UPLOAD_MB` | `50` | Maximum allowed upload size per file |
| `SUMMARY_BATCH_BUDGET`| `12000` | Max character budget per summarization batch |
| `LOG_DOCUMENT_TEXT` | `false` | Set to false to avoid logging private user data |

---

## ⚠️ Known Limitations & Release Boundaries

- **Scanned / Image-Only PDFs:** Pure image scans lacking embedded text streams are flagged with a low-text warning (`DOC-05`). Optical Character Recognition (OCR via Tesseract/PyMuPDF) is slated for Release 2 (`DOC-12`).
- **Session Persistence:** Document vectors are maintained in isolated in-memory session state. Refreshing the browser requires re-processing files. Permanent cross-session database persistence is slated for Release 2 (`DOC-11`).
