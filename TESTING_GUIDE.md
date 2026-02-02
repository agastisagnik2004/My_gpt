# 🧪 Testing Guide - Smart RAG System (LLM → RAG → LLM)

## Quick Start

### 1. Start the Server
```bash
cd /home/whizsoulthree/Documents/RAG_Architecture/Master
python skreach.py
```
Server runs at: `http://127.0.0.1:60922`

### 2. Open Web UI
Visit: http://127.0.0.1:60922

---

## Test Case 1: Intent-Based Context Matching

### Setup: Teach 3 Nandigram Topics
```bash
# Topic 1: General Nandigram
curl -X POST http://127.0.0.1:60922/learn \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Nandigram",
    "description": "Nandigram is a census town in Purba Medinipur district of West Bengal, India. It is known for the 2007 protests against land acquisition."
  }'

# Topic 2: Nandigram Bus Stand
curl -X POST http://127.0.0.1:60922/learn \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Nandigram bus stand",
    "description": "The bus stand is located in the central market area of Nandigram. It serves both local and inter-city buses to Kolkata, Digha, and nearby districts."
  }'

# Topic 3: Nandigram Rail Station
curl -X POST http://127.0.0.1:60922/learn \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Nandigram rail station",
    "description": "The rail station of Nandigram is situated on the Eastern Railway line. It handles both passenger trains and freight trains for long-distance connectivity."
  }'
```

### Test Queries
```bash
# Test 1: "Nandigram bus stand" → Should return BUS info
curl -s -X POST http://127.0.0.1:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Nandigram bus stand"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Intent:', d.get('pipeline_stages',{}).get('intent',{}))
print('Answer:', d.get('answer','')[:150])
"

# Test 2: "Nandigram rail station" → Should return RAIL info
curl -s -X POST http://127.0.0.1:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Nandigram rail station"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Intent:', d.get('pipeline_stages',{}).get('intent',{}))
print('Answer:', d.get('answer','')[:150])
"

# Test 3: "Nandigram rail stand" → Should return RAIL info (not bus!)
curl -s -X POST http://127.0.0.1:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Nandigram rail stand"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Intent:', d.get('pipeline_stages',{}).get('intent',{}))
print('Answer:', d.get('answer','')[:150])
"

# Test 4: "Nandigram" → Should return general info
curl -s -X POST http://127.0.0.1:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Nandigram"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Intent:', d.get('pipeline_stages',{}).get('intent',{}))
print('Answer:', d.get('answer','')[:150])
"
```

### Expected Results

| Query | Intent search_terms | Expected Match | Status |
|-------|---------------------|----------------|--------|
| "Nandigram bus stand" | `["nandigram", "bus", "stand"]` | Bus stand info | ✅ |
| "Nandigram rail station" | `["nandigram", "rail", "station"]` | Rail station info | ✅ |
| "Nandigram rail stand" | `["nandigram", "rail", "stand"]` | Rail station info | ✅ |
| "Nandigram" | `["nandigram"]` | General Nandigram | ✅ |

---

## Test Case 2: Text vs Vector Embedding

### Setup
```bash
# Teach Text Embedding
curl -X POST http://127.0.0.1:60922/learn \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "text embedding",
    "description": "Text embedding converts text into numerical vectors. Word2Vec and GloVe are popular text embedding techniques that map words to continuous vector spaces."
  }'

# Teach Vector Embedding
curl -X POST http://127.0.0.1:60922/learn \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "vector embedding",
    "description": "Vector embedding is the general process of converting high-dimensional data into lower-dimensional vectors. Used for images, audio, text, and more."
  }'
```

### Test Queries
```bash
# Should return TEXT embedding info
curl -s -X POST http://127.0.0.1:60922/query \
  -d '{"question": "text embedding"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print('Answer:', d.get('answer','')[:200])
"

# Should return VECTOR embedding info
curl -s -X POST http://127.0.0.1:60922/query \
  -d '{"question": "vector embedding"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print('Answer:', d.get('answer','')[:200])
"
```

---

## Test Case 3: Verify Intent Extraction

Check that the system correctly extracts intent types:

```bash
# Definition question
curl -s -X POST http://127.0.0.1:60922/query \
  -d '{"question": "What is machine learning?"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print('Intent Type:', d.get('pipeline_stages',{}).get('intent',{}).get('intent_type'))
"
# Expected: "definition"

# How-to question
curl -s -X POST http://127.0.0.1:60922/query \
  -d '{"question": "How to prevent overfitting?"}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
print('Intent Type:', d.get('pipeline_stages',{}).get('intent',{}).get('intent_type'))
"
# Expected: "how_to"
```

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the system (LLM→RAG→LLM) |
| `/learn` | POST | Teach a new topic |
| `/learned-topics` | GET | View all learned topics |
| `/faiss-stats` | GET | FAISS vector store statistics |
| `/health` | GET | System health check |
| `/documents` | GET | List all documents |
| `/cache` | DELETE | Clear response cache |

---

## Debugging Commands

### View Learned Topics
```bash
curl -s http://127.0.0.1:60922/learned-topics | python3 -m json.tool
```

### Check FAISS Stats
```bash
curl -s http://127.0.0.1:60922/faiss-stats | python3 -m json.tool
```

### Clear Cache
```bash
curl -X DELETE http://127.0.0.1:60922/cache
```

### Full Query with All Details
```bash
curl -s -X POST http://127.0.0.1:60922/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Nandigram rail stand"}' | python3 -m json.tool
```

---

## Verification Checklist

### Intent Extraction
- [ ] `intent.search_terms` contains meaningful keywords (no stopwords)
- [ ] `intent.topic` reflects the main subject
- [ ] `intent.intent_type` correctly identifies question type

### Search Quality
- [ ] Bus queries return bus info only
- [ ] Rail queries return rail info only
- [ ] "rail stand" returns rail info (not bus!)
- [ ] General queries return general info

### Response Format
- [ ] `pipeline_stages.intent` shows extracted intent
- [ ] `pipeline_stages.intent_extraction_ms` shows LLM timing
- [ ] `pipeline_stages.retrieval_ms` shows RAG timing

---

## Performance Benchmarks

| Stage | Expected Time |
|-------|---------------|
| Intent Extraction | ~20-30ms |
| RAG Search | ~10-20ms |
| Context Building | ~1ms |
| LLM Response | ~40-60ms |
| **Total** | **~80-120ms** |

---

## Troubleshooting

### Problem: Wrong context returned
**Solution:** Ensure topics are taught separately with specific names

### Problem: "I don't know" response
**Solution:** Check if topic is taught using `/learned-topics`

### Problem: Slow response
**Solution:** Clear cache with `DELETE /cache`

### Problem: Server won't start
**Solution:** Check if port 60922 is in use: `lsof -i :60922`
