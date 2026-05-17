<div align="center">

# Semantic Recommendation Engine

### RAG-Powered Intelligent Product Discovery

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-1C3C3C?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=for-the-badge)](https://www.trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)

A semantic recommendation system that uses **Retrieval-Augmented Generation (RAG)** to deliver intelligent, context-aware product suggestions. Users describe what they're looking for in natural language, and the system retrieves the most relevant items using vector similarity search, metadata filtering, and emotion-based re-ranking.

</div>

---

## How It Works

This engine combines **vector embeddings**, **zero-shot classification**, and **emotion analysis** to go beyond keyword matching and understand the *intent* behind a user's query.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw Product Data                                                  │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│   │  Data Cleaning   │───▶│  Classification   │───▶│   Emotion    │  │
│   │  & Preprocessing │    │  (Zero-Shot NLI)  │    │   Analysis   │  │
│   └─────────────────┘    └──────────────────┘    └──────────────┘  │
│        │                         │                      │          │
│        ▼                         ▼                      ▼          │
│   Cleaned Metadata         Category Labels        Emotion Scores   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              Vector Embedding & Indexing                     │  │
│   │         (Sentence Transformers + ChromaDB)                   │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RUNTIME QUERY FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User Query: "a story about redemption and second chances"         │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────────┐     ┌──────────────┐     ┌───────────────────┐  │
│   │    Embed      │────▶│  Vector Search │────▶│  Retrieve Top 200 │  │
│   │   Query       │     │  (ChromaDB)    │     │  Candidates       │  │
│   └──────────────┘     └──────────────┘     └───────────────────┘  │
│                                                     │              │
│                                                     ▼              │
│                                            ┌──────────────────┐   │
│                                            │  Category Filter  │   │
│                                            │  (Fiction/Nonfic) │   │
│                                            └──────────────────┘   │
│                                                     │              │
│                                                     ▼              │
│                                            ┌──────────────────┐   │
│                                            │  Emotion Re-Rank  │   │
│                                            │  (Joy/Fear/Sad..) │   │
│                                            └──────────────────┘   │
│                                                     │              │
│                                                     ▼              │
│                                            ┌──────────────────┐   │
│                                            │  Top 16 Results   │   │
│                                            │  with Thumbnails  │   │
│                                            └──────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Semantic Search** — Understands natural language queries, not just keywords. "A story about forgiveness" matches books about redemption, atonement, and reconciliation.
- **Category Filtering** — Zero-shot classification using BART-large-MNLI automatically categorizes items into Fiction, Nonfiction, and Children's categories.
- **Emotion-Based Re-Ranking** — Each item is scored across 7 emotions (joy, anger, disgust, fear, sadness, surprise, neutral) using a fine-tuned DistilRoBERTa model. Users can filter by emotional tone.
- **Vector Persistence** — ChromaDB stores embeddings on disk for instant startup after initial indexing.
- **Interactive UI** — Clean Gradio web interface with category dropdowns, tone selectors, and a visual gallery of results.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim sentence transformers for semantic vector representation |
| **Vector Store** | ChromaDB | Persistent vector similarity search with HNSW indexing |
| **Classification** | `facebook/bart-large-mnli` | Zero-shot category assignment via natural language inference |
| **Emotion Detection** | `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion classification on product descriptions |
| **Orchestration** | LangChain | Document loading, text splitting, and embedding pipeline |
| **Frontend** | Gradio (Glass theme) | Interactive web dashboard |
| **Data Processing** | Pandas, NumPy | Feature engineering and data manipulation |

## Project Structure

```
.
├── Gardio-Dashboard.py              # Main application — Gradio web UI
├── Data-explore-Books.ipynb         # Step 1: Data cleaning & exploration
├── Text-classification.ipynb        # Step 2: Zero-shot category classification
├── sentiment-analysis.ipynb         # Step 3: Emotion scoring pipeline
├── Vector-search.ipynb              # Step 4: Embedding generation & vector store setup
├── Platform-products.ipynb          # E-commerce product analysis (extension)
│
├── books_cleaned.csv                # Cleaned dataset (5,197 items)
├── books_final_categorized.csv      # Items with category labels
├── books_with_emotions.csv          # Items with 7 emotion scores
├── tagged_description.txt           # Indexed descriptions for vector store
├── chroma_db/                       # Persisted ChromaDB vector store
├── cover-not-found.jpg              # Fallback thumbnail
│
├── .env                             # Environment variables
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/semantic-recommendation-engine.git
cd semantic-recommendation-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community langchain-text-splitters langchain-chroma \
            langchain-huggingface transformers sentence-transformers \
            pandas numpy gradio python-dotenv kagglehub
```

### Run the Pipeline

Execute the notebooks in order to prepare the data and vector store:

```bash
jupyter notebook
```

1. **Data-explore-Books.ipynb** — Cleans raw data, filters short descriptions, outputs `books_cleaned.csv`
2. **Text-classification.ipynb** — Classifies items into categories, outputs `books_final_categorized.csv`
3. **sentiment-analysis.ipynb** — Generates emotion scores, outputs `books_with_emotions.csv`
4. **Vector-search.ipynb** — Creates embeddings and builds the ChromaDB vector store

### Launch the App

```bash
python Gardio-Dashboard.py
```

The Gradio dashboard will open at `http://localhost:7860`.

## How the RAG Pipeline Works

### 1. Embedding & Indexing

Product descriptions are prepended with their unique ID, split into chunks, and embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`. These are stored in ChromaDB with an HNSW index for fast approximate nearest-neighbor search.

### 2. Semantic Retrieval

When a user submits a query, it's embedded using the same model. ChromaDB returns the top 200 most similar items by cosine distance.

### 3. Metadata Filtering

Results are filtered by category (Fiction, Nonfiction, etc.) and narrowed to the top 16 candidates.

### 4. Emotion Re-Ranking

Pre-computed emotion scores (joy, surprise, anger, fear, sadness) are used to sort results by the user's desired emotional tone. This surfaces items that are not just semantically relevant but emotionally aligned.

## Customization

This system is designed to be adaptable to any product domain:

| Component | What to Change |
|-----------|---------------|
| **Dataset** | Replace `books_with_emotions.csv` with your product catalog |
| **Categories** | Update the zero-shot labels in the classification notebook |
| **Emotions** | Adjust the emotion model or scoring logic in the sentiment notebook |
| **UI** | Modify `Gardio-Dashboard.py` to change layout, filters, or display format |
| **Embedding Model** | Swap `all-MiniLM-L6-v2` for a domain-specific model in the embedding config |

## Acknowledgments

- [LangChain](https://langchain.com) for the RAG orchestration framework
- [HuggingFace](https://huggingface.co) for pre-trained transformer models
- [ChromaDB](https://www.trychroma.com) for the vector database
- [Gradio](https://gradio.app) for the interactive web interface

---

<div align="center">

**Built with RAG, powered by semantic understanding.**

</div>
