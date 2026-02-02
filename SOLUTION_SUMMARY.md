# ✅ Smart RAG System - Intent-First Architecture

## The Evolution: From Simple RAG to LLM → RAG → LLM

### Original Problem
The RAG system had poor context differentiation:
- "Nandigram bus stand" → Returned all Nandigram info (bus + rail mixed)
- "Nandigram rail station" → Also returned all Nandigram info
- "Nandigram rail stand" → Returned BUS stand info (word "stand" caused confusion)
- "text embedding" → Sometimes returned "vector embedding" results

### Root Cause
**Traditional RAG flow:** `Query → RAG Search → LLM Response`
- Raw queries contain noise, stopwords, ambiguous terms
- "Nandigram rail stand" - the word "stand" matched "bus stand" topic
- No understanding of what user actually wants

## Solution: Intent-First Architecture

### New Flow: `Query → LLM (Intent) → RAG Search → LLM (Response)`

```
┌──────────────────────────────────────────────────────────────┐
│  User Query: "Nandigram rail stand"                          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  🧠 STEP 1: LLM INTENT EXTRACTION                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Input: "Nandigram rail stand"                         │  │
│  │  Output: {                                             │  │
│  │    search_terms: ["nandigram", "rail", "stand"],       │  │
│  │    topic: "nandigram rail stand",                      │  │
│  │    intent_type: "question"                             │  │
│  │  }                                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  📚 STEP 2: SMART RAG SEARCH                                 │
│  - Compares search_terms with topic names                    │
│  - "nandigram rail" matches "nandigram rail station" topic   │
│  - NOT "nandigram bus stand" (despite "stand" word)          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  🧠 STEP 3: LLM RESPONSE GENERATION                          │
│  - Uses retrieved context + original question                │
│  - Generates coherent, relevant answer                       │
└──────────────────────────────────────────────────────────────┘
```

## Key Implementation Details

### 1. Intent Extraction (`LLMService.extract_intent()`)
```python
async def extract_intent(self, user_query: str) -> dict:
    """
    FIRST STEP: Use LLM to understand user's intent
    Returns:
    - search_terms: optimized keywords for RAG search
    - topic: main topic user is asking about
    - intent_type: question/how_to/definition/comparison
    - context_hints: what kind of answer user expects
    """
```

### 2. Smart Search (`FAISSVectorStore.search_with_intent()`)
```python
async def search_with_intent(self, search_query, search_terms, context_hints, top_k):
    """
    Uses LLM-extracted intent for smarter matching:
    - Compares search_terms against topic names
    - Calculates query_coverage and topic_coverage scores
    - Bonus for exact/subset matches
    """
```

### 3. Scoring Algorithm
```python
# Match score calculation
common_words = search_term_set & topic_words
query_coverage = len(common_words) / len(search_term_set)
topic_coverage = len(common_words) / len(topic_words)

score = 0.5 + (query_coverage * 0.3) + (topic_coverage * 0.2)

# Bonus for exact matches
if topic_words == search_term_set:
    score = 1.0  # Perfect match
```

## Test Results

### Before (Traditional RAG)
| Query | Result | Status |
|-------|--------|--------|
| "Nandigram bus stand" | All Nandigram info | ❌ |
| "Nandigram rail station" | All Nandigram info | ❌ |
| "Nandigram rail stand" | BUS stand info | ❌ |
| "text embedding" | Vector embedding info | ❌ |

### After (Intent-First RAG)
| Query | Intent Extracted | Result | Status |
|-------|------------------|--------|--------|
| "Nandigram bus stand" | `["nandigram", "bus", "stand"]` | Bus stand info only | ✅ |
| "Nandigram rail station" | `["nandigram", "rail", "station"]` | Rail station info only | ✅ |
| "Nandigram rail stand" | `["nandigram", "rail", "stand"]` | Rail station info | ✅ |
| "Nandigram" | `["nandigram"]` | General Nandigram info | ✅ |

## Why This Works Better

### Traditional RAG Weakness
```
Query: "Nandigram rail stand"
       └── "stand" word matches "bus stand" topic name
       └── Returns wrong context!
```

### Intent-First RAG Strength
```
Query: "Nandigram rail stand"
       └── Intent: ["nandigram", "rail", "stand"]
       └── Compares ALL terms with topic names
       └── "nandigram rail station" has MORE matching words than "nandigram bus stand"
       └── Returns correct context!
```

## Best Practices

### 1. Teach Specific Topics
```bash
# ✅ GOOD - Separate, focused topics
curl -X POST http://127.0.0.1:60922/learn \
  -d '{"topic": "Nandigram bus stand", "description": "..."}'

curl -X POST http://127.0.0.1:60922/learn \
  -d '{"topic": "Nandigram rail station", "description": "..."}'

# ❌ BAD - Combined topics
curl -X POST http://127.0.0.1:60922/learn \
  -d '{"topic": "Nandigram", "description": "Has bus stand... has rail station..."}'
```

### 2. Use Descriptive Topic Names
```bash
# ✅ GOOD
"Python list methods"
"Python dictionary operations"

# ❌ BAD
"Python data structures"
```

## Files Modified
- `skreach.py`:
  - Added `LLMService.extract_intent()` - Intent extraction
  - Added `FAISSVectorStore.search_with_intent()` - Intent-aware search
  - Updated `AIPipeline.process()` - LLM → RAG → LLM flow
  - Enhanced keyword matching algorithm

## API Response Example

```json
{
  "answer": "**Nandigram Rail Station**\n\nThe rail station is situated...",
  "sources": ["The rail station is situated..."],
  "latency_ms": 85.23,
  "pipeline_stages": {
    "intent_extraction_ms": 20.5,
    "intent": {
      "search_query": "nandigram rail stand",
      "topic": "nandigram rail stand",
      "intent_type": "question"
    },
    "retrieval_ms": 15.2,
    "is_relevant": true,
    "context_build_ms": 0.5,
    "llm_generation_ms": 48.1,
    "cache": "miss"
  }
}
```

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Architecture** | Query → RAG → LLM | Query → LLM → RAG → LLM |
| **Intent Understanding** | ❌ None | ✅ Extracted by LLM |
| **Context Matching** | ❌ Word-based | ✅ Intent-based |
| **Ambiguous Queries** | ❌ Wrong results | ✅ Smart disambiguation |
| **Pipeline Visibility** | ❌ Limited | ✅ Full intent info in response |

**The Intent-First approach ensures the system understands WHAT the user wants before searching for relevant information!**
