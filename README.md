# 🤖 Smart RAG Architecture (LLM → RAG → LLM)

A intelligent **Retrieval-Augmented Generation (RAG)** pipeline with **Intent-First Design**. Unlike traditional RAG systems, this uses LLM to understand user intent BEFORE searching, resulting in much better context matching.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 What Makes This Different?

**Traditional RAG:** `Query → RAG Search → LLM Response`
- Problem: Raw queries often contain noise, ambiguity
- "Nandigram rail stand" might match "bus stand" because of word "stand"

**This System:** `Query → LLM (Intent) → RAG Search → LLM Response`
- LLM first extracts user intent and optimized search terms
- "Nandigram rail stand" → Intent: `{topic: "nandigram rail", search_terms: ["nandigram", "rail", "stand"]}`
- RAG searches with understood intent, not raw query

## 📌 Features

- ✅ **Intent-First Search** - LLM understands what user wants BEFORE searching
- ✅ **FAISS Vector Store** - Fast similarity search with 64-dimension embeddings  
- ✅ **Dynamic Learning** - Teach the system new topics on-the-fly
- ✅ **Smart Context Matching** - Differentiates "bus stand" from "rail station" for same location
- ✅ **Async Pipeline** - Fully asynchronous processing
- ✅ **Response Caching** - MD5-based caching for repeated queries
- ✅ **Persistent Storage** - Learned topics saved to disk
- ✅ **Beautiful Web UI** - Chat interface for easy interaction

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│              "Tell me about Nandigram rail stand"               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     1. Cache Check                              │
│                   (MD5 hash lookup)                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠 2. LLM INTENT EXTRACTION (NEW!)                 │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  Input: "Tell me about Nandigram rail stand"        │     │
│     │  Output: {                                          │     │
│     │    search_terms: ["nandigram", "rail", "stand"],    │     │
│     │    topic: "nandigram rail stand",                   │     │
│     │    intent_type: "question"                          │     │
│     │  }                                                  │     │
│     └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. SMART RAG SEARCH                           │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  Uses extracted search_terms to find best match     │     │
│     │  Matches "nandigram rail station" topic (not bus!)  │     │
│     └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. Context Building                           │
│            (Rank & Format Retrieved Docs)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠 5. LLM RESPONSE GENERATION                      │
│              (Generate answer from context)                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline Response                           │
│    (Answer + Sources + Intent Info + Latency Metrics)           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/agastisagnik2004/My_Small_RAG_Architecture.git
cd My_Small_RAG_Architecture

# Install dependencies
pip install fastapi uvicorn pydantic numpy faiss-cpu

# Run the server
python skreach.py
```

The server will start at `http://localhost:60922`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns system status, model info, and indexed document count.

### Query Pipeline
```bash
POST /query
Content-Type: application/json

{
    "question": "What is overfitting?",
    "top_k": 3
}
```

**Response:**
```json
{
    "answer": "**Overfitting**\n\nOverfitting occurs when a model learns the training data too well...",
    "sources": ["Overfitting occurs when a model learns...", "..."],
    "latency_ms": 175.23,
    "pipeline_stages": {
        "intent_extraction_ms": 20.5,
        "intent": {
            "search_query": "overfitting",
            "topic": "overfitting",
            "intent_type": "definition"
        },
        "retrieval_ms": 22.5,
        "context_build_ms": 0.02,
        "llm_generation_ms": 152.1,
        "postprocessing_ms": 0.01,
        "cache": "miss"
    }
}
```

### Teach New Topic 🎓
```bash
POST /learn
Content-Type: application/json

{
    "topic": "Nandigram bus stand",
    "description": "Nandigram bus stand is located in the heart of Nandigram town. It serves as the main bus terminal with routes to Kolkata, Digha, and nearby districts."
}
```

### View Learned Topics
```bash
GET /learned-topics
```

### Add Document
```bash
POST /documents
Content-Type: application/json

{
    "id": "custom_doc_1",
    "content": "Your custom knowledge content here",
    "metadata": {"category": "custom"}
}
```

### List Documents
```bash
GET /documents
```

### FAISS Statistics
```bash
GET /faiss-stats
```

### Clear Cache
```bash
DELETE /cache
```

## 🧠 Dynamic Learning

The system can learn new topics on-the-fly! Here's how:

```bash
# Step 1: Ask about something unknown
curl -X POST http://localhost:60922/query \
  -d '{"question": "What is Nandigram bus stand?"}'
# Response: "I don't know about this. Please teach me!"

# Step 2: Teach the system
curl -X POST http://localhost:60922/learn \
  -d '{
    "topic": "Nandigram bus stand",
    "description": "Nandigram bus stand is the main bus terminal..."
  }'

# Step 3: Now it knows!
curl -X POST http://localhost:60922/query \
  -d '{"question": "Tell me about Nandigram bus stand"}'
# Response: Returns the learned information!
```

## 📚 Pre-loaded Knowledge Base

The system comes with 25+ documents covering:

| Category | Topics |
|----------|--------|
| **ML Basics** | Machine Learning, Supervised Learning, Unsupervised Learning |
| **Neural Networks** | Architecture, Deep Learning, Activation Functions |
| **Training** | Gradient Descent, Backpropagation, Epochs, Batches |
| **Regularization** | Overfitting, Underfitting, Dropout, L1/L2 |
| **Evaluation** | Accuracy, Precision, Recall, Cross-Validation |
| **Data Processing** | Preprocessing, Feature Engineering, Normalization |
| **Advanced** | Transfer Learning, Ensemble Methods, Hyperparameter Tuning |

## 🧪 Example Queries

```bash
# Basic query
curl -X POST http://localhost:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?"}'

# Context-specific query (intent-aware!)
curl -X POST http://localhost:60922/query \
  -d '{"question": "Nandigram rail stand"}'
# Will correctly match "Nandigram rail station" not "bus stand"!

# Learning flow
curl -X POST http://localhost:60922/learn \
  -d '{"topic": "My custom topic", "description": "Detailed explanation..."}'
```

## 🔧 Project Structure

```
My_Small_RAG_Architecture/
├── skreach.py           # Main application (LLM→RAG→LLM pipeline)
├── faiss_metadata.json  # Persisted learned topics & queries
├── README.md            # Documentation
├── static/
│   ├── css/styles.css   # Web UI styles
│   └── js/script.js     # Web UI logic
└── templates/
    └── index.html       # Chat interface
```

## 🎯 Key Components

| Component | Description |
|-----------|-------------|
| `FAISSVectorStore` | FAISS-based vector database with intent-aware search |
| `LLMService` | Intent extraction + response generation |
| `AIPipeline` | Orchestrator: LLM → RAG → LLM flow |
| `FastAPI App` | RESTful API + Web UI |

## 🔑 Why Intent-First Works Better

| Query | Traditional RAG | Intent-First RAG |
|-------|-----------------|------------------|
| "Nandigram rail stand" | Might match "bus stand" (word "stand") | ✅ Matches "rail station" |
| "text Embedding" | Might match "vector embedding" | ✅ Matches only "text embedding" |
| "How to train model?" | Generic search | ✅ Knows intent is "how_to" |

## 📖 API Documentation

Once the server is running, visit:
- **Web UI**: http://localhost:60922
- **Swagger UI**: http://localhost:60922/docs
- **ReDoc**: http://localhost:60922/redoc

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Sagnik Agasti**
- GitHub: [@agastisagnik2004](https://github.com/agastisagnik2004)

---

⭐ If you found this helpful, please give it a star!
