# UniAssist+ 🤖
### A Tool-Augmented Multi-Agent RAG System for Intelligent University Support
**Atom Camp  •  Data Science & AI Bootcamp  •  Final Project  •  2025–2026**

---

## What is UniAssist+?

UniAssist+ is an intelligent AI assistant for KIET University students and staff. Unlike a basic chatbot, it combines a full **Retrieval-Augmented Generation (RAG) pipeline** with an **action-oriented tool layer** — enabling it to both answer information queries and perform structured tasks like drafting emails and generating official documents.

---

## Pipeline Modes

| Mode | Trigger | What Happens |
|------|---------|--------------|
| **RAG** | Information query | FAQ → Vector DB → Web Scraper → LLM → Consistency Check → Summarize |
| **ACTION** | Email / document / academic info request | Route directly to the matching tool |
| **HYBRID** | Action + context keywords together | Retrieve RAG context first, then invoke tool with enriched request |

---

## Agents & Tools

**RAG Agents**
- `FAQ Agent` — MongoDB fuzzy matching with thread-safe caching and smalltalk detection
- `Vector DB Agent` — FAISS semantic search (E5-base-v2 embeddings) over university documents
- `Web Scraper Agent` — Real-time KIET website scraping (BeautifulSoup + Playwright)
- `LLM Agent` — Llama 3.2 1B Instruct (HuggingFace), CPU-optimized inference
- `Consistency Check Agent` — Hallucination detection via fuzzy overlap + optional LLM self-verify
- `Summarizing Agent` — Student-friendly bullet-point output

**Tool Layer**
- `Email Draft Tool` — Drafts formal emails (leave, complaints, applications, professor queries)
- `Document Generator Tool` — NOC, Bonafide Certificate, Internship Letter, Leave Application
- `Academic Info Tool` — Simulated university API: programs, fees, grading, calendar, facilities

**Infrastructure**
- `Response Log Agent` — Non-blocking MongoDB logger for all interactions
- `Orchestrator Agent` — Central planner: intent classification + routing + link appending + FAQ learning

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Vector Database | FAISS (via LangChain) |
| Embeddings | intfloat/e5-base-v2 (Sentence Transformers) |
| LLM | Llama 3.2 1B Instruct (HuggingFace) |
| Database | MongoDB (PyMongo) |
| Web Scraping | BeautifulSoup4 + crawl4ai + Playwright |
| Fuzzy Matching | RapidFuzz |
| PDF Extraction | PyMuPDF (fitz) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Config | python-dotenv |

---

## Project Structure

```
FINAL_PROJECT/
├── agents/
│   ├── orchestrator_agent.py      # Central planner + intent routing
│   ├── faq_agent.py               # MongoDB FAQ retrieval
│   ├── vectordb_agent.py          # FAISS semantic search
│   ├── webscraper_agent.py        # Real-time KIET website scraper
│   ├── llm_agent.py               # HuggingFace LLM inference
│   ├── consistency_check_agent.py # Hallucination guard
│   └── summarizing_agent.py       # Bullet-point summarizer
│
├── tools/
│   ├── email_draft_tool.py        # Email drafting tool
│   ├── document_generator_tool.py # Document generation tool
│   ├── academic_info_tool.py      # Simulated university API
│   └── response_log_agent.py      # MongoDB response logger
│
├── config/
│   ├── settings.py                # .env config + startup validation
│   └── urls.txt                   # Web scraper seed URLs
│
├── pages/
│   ├── 1_🤖_Chatbot.py            # Main chat interface
│   └── 2_🌐_EDU_Assist_Website.py # Website-style view
│
├── utils/
│   └── ui_helpers.py              # Shared UI utilities
│
├── data/
│   ├── documents/                 # University PDFs and TXTs (for FAISS)
│   └── vectorstore/               # FAISS index (auto-generated, gitignored)
│
├── Home.py                        # Streamlit entry point
├── pipeline.py                    # Singleton bot builder
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd FINAL_PROJECT
pip install -r requirements.txt

# Optional: dynamic web scraping
pip install playwright && playwright install chromium
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and fill in:
#   MONGO_URI=mongodb://localhost:27017
#   HUGGINGFACE_TOKEN=hf_...
```

### 3. Add Data
- Place university PDF/TXT files in `data/documents/`
- Add KIET page URLs to `config/urls.txt` (one per line)
- Load FAQ data into MongoDB: `FAQ_AGENT.FAQs` collection with `question` and `answer` fields

### 4. Run
```bash
streamlit run Home.py
```

---

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `MONGO_URI` | `mongodb://localhost:27017` | Yes |
| `DB_NAME` | `FAQ_AGENT` | No |
| `HUGGINGFACE_TOKEN` | — | Yes (for Llama) |
| `LLM_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | No |
| `LLM_MAX_TOKENS` | `256` | No |
| `DOCUMENT_FOLDER` | `data/documents` | No |
| `VECTOR_DB_FOLDER` | `data/vectorstore` | No |
| `URLS_FILE` | `config/urls.txt` | No |

---

## Important Notes

- **Never commit `.env`** — it contains your HuggingFace token and MongoDB URI
- **FAISS index** is auto-generated on first run — do not commit `data/vectorstore/`
- **Minimum 8 GB RAM** recommended for CPU-based LLM inference
- The system uses **Llama 3.2 1B** — a gated HuggingFace model. You must accept the license on HuggingFace before using it.

---

*UniAssist+ — Atom Camp  •  Data Science & AI Bootcamp  •  Final Project*
