# 🎬 Mood-Based Movie Recommendation Assistant

A RAG (Retrieval Augmented Generation) system that recommends movies based on natural language queries like:
- *"Something comforting after a stressful day"*
- *"A funny movie for a family dinner"*
- *"I need something that doesn't require much attention"*

## Tech Stack
- **Python** — core language
- **sentence-transformers** — semantic text embeddings
- **FAISS** — vector similarity search
- **Streamlit** — user interface *(in progress)*
- **Claude API** — natural language response generation *(in progress)*

## Dataset
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — enriched with custom mood-based tags.

## Custom Enrichment Tags
| Tag | Values |
|---|---|
| `mood` | happy, dark, romantic, tense, exciting, whimsical, emotional, thoughtful, neutral |
| `pacing` | slow, moderate, fast |
| `attention_required` | low, medium, high |
| `emotional_intensity` | low, medium, high |
| `family_friendly` | yes, no |
| `watch_with_food` | yes, no |

## Project Structure
movie-rag/
├── data/ # Dataset files (not tracked in git)
├── notebooks/
│ ├── 01_explore_and_clean.py # Data cleaning
│ ├── 02_enrich.py # Custom tag enrichment
│ ├── 03_embeddings.py # Sentence-transformer embeddings
│ ├── 04_build_faiss_index.py # FAISS index construction
│ └── 05_search.py # Semantic search function
└── README.md

## Setup
```bash
# Install dependencies
pip install pandas numpy sentence-transformers faiss-cpu streamlit

# Download TMDB dataset from Kaggle and place CSVs in data/

# Run pipeline in order
python notebooks/01_explore_and_clean.py
python notebooks/02_enrich.py
python notebooks/03_embeddings.py
python notebooks/04_build_faiss_index.py
python notebooks/05_search.py
```

## Status
- [x] Data cleaning & preprocessing
- [x] Custom mood-based enrichment
- [x] Semantic embeddings (all-MiniLM-L6-v2)
- [x] FAISS vector index
- [x] Semantic search function
- [ ] Claude API integration
- [ ] Streamlit UI