"""
End-to-End AI Pipeline Demo
A small but complete RAG (Retrieval-Augmented Generation) pipeline
"""

import asyncio
import time
import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Pipeline", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== Data Models ==============

class Document(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)

class PipelineResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    pipeline_stages: dict

# ============== FAISS Vector Database ==============

@dataclass
class VectorEntry:
    doc_id: str
    content: str
    embedding: list[float]
    metadata: dict

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    EMBEDDING_DIM = 64  # Dimension of our embeddings
    PERSISTENCE_FILE = "faiss_index.bin"
    METADATA_FILE = "faiss_metadata.json"
    RELEVANCE_THRESHOLD = 0.15  # Minimum similarity score to consider relevant
    
    def __init__(self):
        self.entries: list[VectorEntry] = []
        self.index: faiss.IndexFlatIP = None  # Inner Product (cosine sim with normalized vectors)
        self.user_queries: list[dict] = []  # Store user queries
        self.learned_topics: dict[str, str] = {}  # Store user-taught topics
        self.pending_topics: dict[str, float] = {}  # Topics waiting to be taught {topic: timestamp}
        self._initialize_faiss()
        self._load_persisted_data()
        self._validate_and_sync_data()  # Sync metadata with FAISS index
        self._initialize_knowledge_base()
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        # Using IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        logger.info(f"FAISS index initialized with dimension {self.EMBEDDING_DIM}")
    
    def _load_persisted_data(self):
        """Load persisted index and metadata if exists"""
        try:
            metadata_path = os.path.join(STATIC_DIR, self.METADATA_FILE)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.user_queries = data.get('user_queries', [])
                    self.learned_topics = data.get('learned_topics', {})
                logger.info(f"Loaded {len(self.user_queries)} persisted user queries and {len(self.learned_topics)} learned topics")
        except Exception as e:
            logger.warning(f"Could not load persisted data: {e}")
    
    def _validate_and_sync_data(self):
        """Validate and sync metadata with FAISS index - handles deleted entries"""
        # Clear FAISS index completely and rebuild from metadata only
        # This ensures if something is deleted from metadata, it's also gone from FAISS
        self._initialize_faiss()  # Reset FAISS index
        self.entries = []  # Clear entries
        
        # Rebuild learned topics from metadata
        self._rebuild_learned_topics()
        
        logger.info(f"Data validated and synced. FAISS index has {self.index.ntotal} entries")
    
    def _rebuild_learned_topics(self):
        """Rebuild learned topics entries in FAISS after loading from disk with chunking"""
        if not self.learned_topics:
            return
        
        embeddings_list = []
        for topic, description in self.learned_topics.items():
            # Use chunking for each learned topic
            chunks = self._chunk_description(description, topic)
            
            for chunk_index, chunk_text in enumerate(chunks):
                embedding = self._compute_embedding(topic + " " + chunk_text)
                embeddings_list.append(embedding)
                chunk_id = hashlib.md5((topic + chunk_text).encode()).hexdigest()[:8]
                doc_id = f"learned_{chunk_id}"
                self.entries.append(VectorEntry(
                    doc_id=doc_id,
                    content=chunk_text,
                    embedding=embedding.tolist(),
                    metadata={
                        "type": "learned",
                        "topic": topic,
                        "keywords": topic + " " + chunk_text.lower(),
                        "full_topic": topic,
                        "chunk_index": chunk_index
                    }
                ))
        
        if embeddings_list:
            embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
            self.index.add(embeddings_matrix)
            logger.info(f"Rebuilt {len(self.learned_topics)} learned topics in FAISS index")
    
    def _save_persisted_data(self):
        """Save metadata to disk"""
        try:
            metadata_path = os.path.join(STATIC_DIR, self.METADATA_FILE)
            with open(metadata_path, 'w') as f:
                json.dump({
                    'user_queries': self.user_queries,
                    'learned_topics': self.learned_topics,
                    'total_entries': len(self.entries)
                }, f, indent=2)
            logger.info(f"Persisted {len(self.user_queries)} user queries and {len(self.learned_topics)} learned topics")
        except Exception as e:
            logger.error(f"Could not save persisted data: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            index_path = os.path.join(STATIC_DIR, self.PERSISTENCE_FILE)
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS index saved to {index_path}")
        except Exception as e:
            logger.error(f"Could not save FAISS index: {e}")
    
    def _initialize_knowledge_base(self):
        """Pre-load comprehensive documents into the vector store"""
        knowledge = [
            # Machine Learning Basics
            ("ml_basics_1", "machine learning", "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions."),
            ("ml_basics_2", "supervised learning", "Supervised learning is a type of machine learning where the model learns from labeled training data. Examples include classification (predicting categories) and regression (predicting continuous values)."),
            ("ml_basics_3", "unsupervised learning", "Unsupervised learning works with unlabeled data to discover hidden patterns. Common techniques include clustering (grouping similar data) and dimensionality reduction (reducing features while preserving information)."),
            
            # Neural Networks
            ("nn_1", "neural network", "A neural network is a computational model inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. Input layer receives data, hidden layers process it, and output layer produces results."),
            ("nn_2", "deep learning", "Deep learning uses neural networks with many hidden layers (deep neural networks) to learn complex patterns. It excels at image recognition, natural language processing, and speech recognition tasks."),
            ("nn_3", "activation function", "Activation functions introduce non-linearity into neural networks. Common ones include ReLU (Rectified Linear Unit), sigmoid, and tanh. They determine whether a neuron should be activated based on input."),
            
            # Training Concepts
            ("train_1", "training data", "Training data is the dataset used to teach a machine learning model. It should be representative of real-world scenarios, properly labeled (for supervised learning), and large enough to capture patterns."),
            ("train_2", "gradient descent", "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively adjusting model parameters in the direction of steepest descent. Learning rate controls the step size."),
            ("train_3", "backpropagation", "Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight, propagating error backwards through the network layers."),
            ("train_4", "epoch batch", "An epoch is one complete pass through the entire training dataset. A batch is a subset of training data processed together. Mini-batch gradient descent updates weights after each batch."),
            
            # Overfitting and Regularization
            ("overfit_1", "overfitting", "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on new unseen data. Signs include high training accuracy but low validation accuracy."),
            ("overfit_2", "underfitting", "Underfitting happens when a model is too simple to capture the underlying patterns in data. It performs poorly on both training and test data. Solutions include using more complex models or adding features."),
            ("reg_1", "regularization", "Regularization techniques prevent overfitting by adding constraints to the model. L1 regularization (Lasso) adds absolute weight penalty, L2 regularization (Ridge) adds squared weight penalty."),
            ("reg_2", "dropout", "Dropout is a regularization technique where random neurons are temporarily removed during training. This prevents neurons from co-adapting and forces the network to learn more robust features."),
            
            # Model Evaluation
            ("eval_1", "accuracy precision recall", "Accuracy measures overall correct predictions. Precision measures how many positive predictions were correct. Recall measures how many actual positives were identified. F1-score balances precision and recall."),
            ("eval_2", "confusion matrix", "A confusion matrix shows true positives, true negatives, false positives, and false negatives. It helps visualize model performance and identify which classes are being confused."),
            ("eval_3", "cross validation", "Cross-validation splits data into k folds, training on k-1 folds and validating on the remaining fold, rotating through all combinations. It provides a more reliable estimate of model performance."),
            ("eval_4", "train test split", "Data should be split into training set (to learn patterns), validation set (to tune hyperparameters), and test set (to evaluate final performance). Common split ratios are 70-15-15 or 80-10-10."),
            
            # Data Processing
            ("data_1", "data preprocessing", "Data preprocessing prepares raw data for modeling. Steps include handling missing values, removing duplicates, encoding categorical variables, and scaling numerical features."),
            ("data_2", "feature engineering", "Feature engineering creates new features from existing data to improve model performance. Techniques include polynomial features, interaction terms, binning, and domain-specific transformations."),
            ("data_3", "normalization scaling", "Normalization scales features to a range (usually 0-1). Standardization transforms data to have zero mean and unit variance. Both help models converge faster and perform better."),
            
            # Advanced Topics
            ("adv_1", "transfer learning", "Transfer learning reuses a pre-trained model on a new task. The model's learned features from one domain are applied to another, reducing training time and data requirements."),
            ("adv_2", "hyperparameter tuning", "Hyperparameters are settings chosen before training (learning rate, batch size, layers). Tuning methods include grid search, random search, and Bayesian optimization."),
            ("adv_3", "ensemble methods", "Ensemble methods combine multiple models for better predictions. Bagging (Random Forest) reduces variance, boosting (XGBoost, AdaBoost) reduces bias, stacking uses a meta-learner."),
            
            # Loss Functions
            ("loss_1", "loss function", "A loss function measures how well the model's predictions match actual values. MSE (Mean Squared Error) for regression, Cross-Entropy for classification. Lower loss indicates better performance."),
            ("loss_2", "optimizer", "Optimizers update model weights to minimize loss. SGD (Stochastic Gradient Descent), Adam (Adaptive Moment Estimation), and RMSprop are popular choices with different convergence properties."),
        ]
        
        # Add all knowledge base entries to FAISS
        embeddings_list = []
        for doc_id, keywords, content in knowledge:
            embedding = self._compute_embedding(keywords + " " + content)
            embeddings_list.append(embedding)
            self.entries.append(VectorEntry(doc_id, content, embedding.tolist(), {"keywords": keywords}))
        
        # Batch add to FAISS index
        if embeddings_list:
            embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
            self.index.add(embeddings_matrix)
            logger.info(f"Added {len(knowledge)} documents to FAISS index. Total: {self.index.ntotal}")
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding using TF-based word vectors for FAISS"""
        # Key terms for ML domain - creates a pseudo-semantic embedding
        key_terms = [
            "machine", "learning", "neural", "network", "deep", "training",
            "data", "model", "layer", "weight", "loss", "gradient", "descent",
            "overfit", "regularization", "dropout", "accuracy", "precision",
            "recall", "validation", "test", "feature", "supervised", "unsupervised",
            "classification", "regression", "cluster", "activation", "backpropagation",
            "epoch", "batch", "optimizer", "transfer", "ensemble", "hyperparameter",
            # Extended terms for better coverage
            "algorithm", "prediction", "error", "bias", "variance", "input",
            "output", "hidden", "neuron", "parameter", "function", "cost",
            "convergence", "iteration", "sample", "distribution", "probability",
            "embedding", "vector", "matrix", "tensor", "dimension", "normalize",
            "threshold", "sigmoid", "softmax", "relu", "pooling", "convolution"
        ]
        # Ensure we have exactly EMBEDDING_DIM terms
        key_terms = key_terms[:self.EMBEDDING_DIM]
        while len(key_terms) < self.EMBEDDING_DIM:
            key_terms.append(f"term_{len(key_terms)}")
        
        text_lower = text.lower()
        # Create embedding based on term frequency
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        for i, term in enumerate(key_terms):
            count = text_lower.count(term)
            embedding[i] = min(count / 3.0, 1.0)  # Normalize
        
        # L2 normalize for cosine similarity with IndexFlatIP
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def add_to_index(self, embedding: np.ndarray):
        """Add a single embedding to FAISS index"""
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding_2d)
    
    def add_user_query(self, query: str):
        """Automatically add new user query to vector database"""
        # Generate unique ID
        query_id = f"user_query_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        # Check if already exists
        for uq in self.user_queries:
            if uq['content'].lower() == query.lower():
                logger.info(f"Query already exists in DB: {query[:30]}...")
                return False
        
        # Compute embedding
        embedding = self._compute_embedding(query)
        
        # Add to FAISS index
        self.add_to_index(embedding)
        
        # Add to entries list
        entry = VectorEntry(
            doc_id=query_id,
            content=f"User asked: {query}",
            embedding=embedding.tolist(),
            metadata={"type": "user_query", "original_query": query}
        )
        self.entries.append(entry)
        
        # Track user query
        self.user_queries.append({
            "id": query_id,
            "content": query,
            "timestamp": time.time()
        })
        
        # Persist to disk
        self._save_persisted_data()
        
        logger.info(f"Added new user query to FAISS: {query[:50]}...")
        return True
    
    async def search_with_intent(self, search_query: str, search_terms: list[str], 
                                  context_hints: dict, top_k: int = 3) -> list[tuple[str, float]]:
        """
        ADAPTIVE SMART SEARCH: Automatically adapts to new learned contexts
        Uses LLM-extracted intent + learns from stored topics for better matching
        """
        results = []
        
        # ADAPTIVE: Build context vocabulary from learned topics
        # This helps the system adapt when new topics are taught
        learned_context_words = set()
        for topic in self.learned_topics.keys():
            learned_context_words.update(topic.split())
        
        # Search learned topics using intent-based matching
        for entry in self.entries:
            if entry.metadata.get('type') != 'learned':
                continue
                
            topic = entry.metadata.get('topic', '').lower()
            content_lower = entry.content.lower()
            
            if not topic:
                continue
            
            # Match search terms against topic name
            topic_words = set(topic.replace('-', ' ').replace('_', ' ').split())
            search_term_set = set(search_terms)
            
            # ADAPTIVE MATCHING: Check both topic name AND content keywords
            # This allows the system to find new contexts even with partial matches
            content_keywords = set(content_lower.split()[:50])  # First 50 words as keywords
            
            # Calculate match score
            topic_common = search_term_set & topic_words
            content_common = search_term_set & content_keywords
            
            if topic_common:
                # Score based on how well query matches topic
                query_coverage = len(topic_common) / len(search_term_set) if search_term_set else 0
                topic_coverage = len(topic_common) / len(topic_words) if topic_words else 0
                
                score = 0.5 + (query_coverage * 0.3) + (topic_coverage * 0.2)
                
                # Bonus for exact or near-exact matches
                if topic_words == search_term_set:
                    score = 1.0  # Perfect match
                elif topic_words.issubset(search_term_set) or search_term_set.issubset(topic_words):
                    score += 0.15
                
                # ADAPTIVE BONUS: Check remaining terms in content
                remaining = search_term_set - topic_words
                if remaining:
                    content_hits = sum(1 for w in remaining if w in content_lower)
                    if content_hits > 0:
                        score += 0.05 * (content_hits / len(remaining))
                
                results.append((entry.content, score))
                
            elif content_common and len(content_common) >= 2:
                # ADAPTIVE: Even without topic name match, find by content keywords
                # Requires at least 2 matching words to avoid false positives
                content_score = 0.3 + (len(content_common) / len(search_term_set)) * 0.3
                results.append((entry.content, content_score))
        
        # Also check knowledge base entries
        for entry in self.entries:
            if entry.metadata.get('type') == 'learned':
                continue
            
            content_lower = entry.content.lower()
            keywords = entry.metadata.get('keywords', '').lower()
            
            matched = sum(1 for w in search_terms if w in content_lower or w in keywords)
            if matched > 0:
                ratio = matched / len(search_terms) if search_terms else 0
                if ratio >= 0.5:
                    score = 0.5 + ratio * 0.35
                    results.append((entry.content, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results and results[0][1] >= 0.25:
            return results[:top_k]
        
        # Fallback to FAISS vector search
        if self.index.ntotal > 0:
            query_embedding = self._compute_embedding(search_query)
            query_embedding_2d = query_embedding.reshape(1, -1).astype(np.float32)
            
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding_2d, k)
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.entries) and idx >= 0:
                    content = self.entries[idx].content
                    score = float(distances[0][i])
                    results.append((content, score))
        
        return results[:top_k] if results else []
    
    async def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Search for similar documents using FAISS with improved context matching"""
        query_lower = query.lower().strip()
        
        # Remove common words to get core topic
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 
                     'could', 'would', 'should', 'in', 'to', 'for', 'and', 'or', 'of',
                     'explain', 'tell', 'me', 'about', 'describe', 'define', 'why',
                     'when', 'where', 'which', 'who', 'please', 'help', 'understand'}
        
        query_words = [w for w in query_lower.replace('?', '').replace(',', '').replace('.', '').split() 
                       if w not in stopwords and len(w) > 1]
        
        # Extract context words (words that indicate a specific aspect)
        # These help differentiate between "bus stand" vs "rail station" for the same place
        context_words = query_words.copy()
        
        # First check if query matches a learned topic directly
        if query_lower in self.learned_topics:
            return [(self.learned_topics[query_lower], 1.0)]  # Perfect match
        
        # Check each word against learned topics
        for word in query_words:
            if word in self.learned_topics:
                # Don't return full description directly - let keyword search handle chunks
                break
        
        # Check partial matches in learned topics with context consideration
        # Skip this section - let keyword_search handle context-specific retrieval from chunks
        learned_topic_matched = False
        for topic, description in self.learned_topics.items():
            if topic in query_lower or query_lower in topic:
                # Topic is partially matched but don't return full description
                # Instead, let keyword search find the appropriate chunk
                learned_topic_matched = True
                break
            # Check if any query word matches topic
            for word in query_words:
                if word in topic or topic in word:
                    learned_topic_matched = True
                    break
        
        # Keyword search in knowledge base entries (now includes chunked learned topics)
        keyword_matches = self._keyword_search(query_lower, query_words)
        if keyword_matches:
            return keyword_matches
        
        # If learned topic was matched but no specific chunks found, return full description
        if learned_topic_matched:
            for topic, description in self.learned_topics.items():
                if topic in query_lower or query_lower in topic or any(w in topic for w in query_words):
                    return [(description, 0.8)]
        
        if self.index.ntotal == 0:
            return []
        
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        query_embedding_2d = query_embedding.reshape(1, -1).astype(np.float32)
        
        # FAISS search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding_2d, k)
        
        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.entries) and idx >= 0:
                content = self.entries[idx].content
                score = float(distances[0][i])  # Already similarity (inner product)
                results.append((content, score))
        
        return results
    
    def _calculate_context_match(self, context_words: list[str], text: str) -> float:
        """Calculate how well the context words match in the text (0 to 1)"""
        text_lower = text.lower()
        matches = sum(1 for word in context_words if word in text_lower)
        return matches / len(context_words) if context_words else 0
    
    def _keyword_search(self, query_lower: str, query_words: list[str]) -> list[tuple[str, float]]:
        """Smart search that understands user intent by matching query words against topic names"""
        results = []
        
        for entry in self.entries:
            content_lower = entry.content.lower()
            keywords = entry.metadata.get('keywords', '').lower()
            topic = entry.metadata.get('topic', '').lower() if entry.metadata.get('topic') else ''
            entry_type = entry.metadata.get('type', '')
            
            score = 0.0
            
            if entry_type == 'learned' and topic:
                # SMART MATCHING: Compare query words with topic name words
                topic_words = set(topic.replace('-', ' ').replace('_', ' ').split())
                query_word_set = set(query_words)
                
                # Find common words between query and topic name
                common_words = query_word_set & topic_words
                
                if common_words:
                    # Calculate match score based on:
                    # 1. How many query words match the topic name
                    # 2. How specific the match is (more common words = better match)
                    
                    query_coverage = len(common_words) / len(query_word_set) if query_word_set else 0
                    topic_coverage = len(common_words) / len(topic_words) if topic_words else 0
                    
                    # Combined score - prioritize topics where more query words match
                    score = 0.5 + (query_coverage * 0.3) + (topic_coverage * 0.2)
                    
                    # BONUS: If topic name is a subset of query or vice versa, boost score
                    if topic_words.issubset(query_word_set) or query_word_set.issubset(topic_words):
                        score += 0.15
                    
                    # BONUS: Check if remaining query words appear in content (intent verification)
                    remaining_words = query_word_set - topic_words
                    if remaining_words:
                        content_matches = sum(1 for w in remaining_words if w in content_lower)
                        if content_matches > 0:
                            score += 0.05 * (content_matches / len(remaining_words))
                else:
                    # No direct topic match - check content for relevance
                    content_matches = sum(1 for w in query_words if w in content_lower or w in keywords)
                    if content_matches > 0:
                        score = 0.3 + (content_matches / len(query_words)) * 0.2 if query_words else 0.3
            else:
                # For knowledge base entries
                if query_lower in content_lower or query_lower in keywords:
                    score = 0.85
                else:
                    matched_words = sum(1 for w in query_words if w in content_lower or w in keywords)
                    if matched_words > 0:
                        match_ratio = matched_words / len(query_words) if query_words else 0
                        if match_ratio >= 0.5:
                            score = 0.5 + match_ratio * 0.35
            
            if score > 0:
                results.append((entry.content, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results if they meet threshold
        if results and results[0][1] >= 0.25:
            return results[:3]
        
        return []
    
    def is_relevant_match(self, results: list[tuple[str, float]]) -> bool:
        """Check if search results are relevant enough"""
        if not results:
            return False
        # Check if the best match exceeds threshold
        best_score = results[0][1] if results else 0
        return best_score >= self.RELEVANCE_THRESHOLD
    
    def learn_topic(self, topic: str, description: str, cache_ref=None):
        """Learn a new topic from user input with automatic chunking for better retrieval"""
        topic_lower = topic.lower().strip()
        
        # Check if topic already exists - update it
        is_update = topic_lower in self.learned_topics
        
        # Store in learned topics
        self.learned_topics[topic_lower] = description
        
        # Split description into context-aware chunks if it's long and contains context markers
        chunks = self._chunk_description(description, topic)
        
        # Add all chunks to FAISS
        for chunk_index, chunk_text in enumerate(chunks):
            embedding = self._compute_embedding(topic + " " + chunk_text)
            self.add_to_index(embedding)
            
            # Add each chunk as a separate entry with metadata
            chunk_id = hashlib.md5((topic + chunk_text).encode()).hexdigest()[:8]
            doc_id = f"learned_{chunk_id}"
            entry = VectorEntry(
                doc_id=doc_id,
                content=chunk_text,
                embedding=embedding.tolist(),
                # Include both topic name and content for better matching
                metadata={
                    "type": "learned",
                    "topic": topic,
                    "keywords": topic_lower + " " + chunk_text.lower(),  # Enhanced keywords with chunk content
                    "full_topic": topic_lower,
                    "chunk_index": chunk_index
                }
            )
            self.entries.append(entry)
        
        # Clear cache for this topic if cache reference provided
        if cache_ref is not None:
            keys_to_remove = []
            for key, response in list(cache_ref.items()):
                # Check if this cache entry might be related to the topic
                if topic_lower in str(response.answer).lower() or "still learning" in response.answer.lower():
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del cache_ref[key]
            if keys_to_remove:
                logger.info(f"Cleared {len(keys_to_remove)} cache entries for topic: {topic}")
        
        # Persist data
        self._save_persisted_data()
        self._save_faiss_index()
        
        action = "Updated" if is_update else "Learned"
        logger.info(f"{action} topic: {topic} with {len(chunks)} context chunks")
        return True
    
    def _chunk_description(self, description: str, topic: str) -> list[str]:
        """Disable automatic chunking - users should teach separate topics for context differentiation"""
        # For better context separation and retrieval accuracy,
        # it's recommended to teach separate, focused topics rather than
        # one large topic with multiple concepts.
        # Example: Teach "Nandigram bus stand" and "Nandigram rail station" separately
        # instead of "Nandigram" with both in description
        return [description]
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "total_entries": len(self.entries),
            "user_queries_count": len(self.user_queries),
            "learned_topics_count": len(self.learned_topics),
            "pending_topics_count": len(self.pending_topics),
            "embedding_dimension": self.EMBEDDING_DIM,
            "relevance_threshold": self.RELEVANCE_THRESHOLD
        }
    
    def set_pending_topic(self, topic: str):
        """Set a topic as pending (waiting for user to teach)"""
        topic_lower = topic.lower().strip()
        self.pending_topics[topic_lower] = time.time()
        logger.info(f"Set pending topic: {topic_lower}")
    
    def get_pending_topic(self) -> str | None:
        """Get the most recent pending topic (within last 5 minutes)"""
        if not self.pending_topics:
            return None
        
        current_time = time.time()
        # Find most recent pending topic within 5 minutes
        valid_topics = [(topic, ts) for topic, ts in self.pending_topics.items() 
                        if current_time - ts < 300]  # 5 minutes
        
        if valid_topics:
            # Return most recent
            valid_topics.sort(key=lambda x: x[1], reverse=True)
            return valid_topics[0][0]
        return None
    
    def clear_pending_topic(self, topic: str):
        """Clear a pending topic after it's been taught"""
        topic_lower = topic.lower().strip()
        if topic_lower in self.pending_topics:
            del self.pending_topics[topic_lower]
            logger.info(f"Cleared pending topic: {topic_lower}")
    
    def delete_topic(self, topic: str) -> bool:
        """Delete a learned topic completely - will require fresh learning"""
        topic_lower = topic.lower().strip()
        
        if topic_lower not in self.learned_topics:
            return False
        
        # Remove from learned_topics
        del self.learned_topics[topic_lower]
        
        # Rebuild entire index to remove the deleted topic
        self._validate_and_sync_data()
        self._initialize_knowledge_base()
        
        # Persist changes
        self._save_persisted_data()
        
        logger.info(f"Deleted topic: {topic} - will learn fresh next time")
        return True
    
    def is_user_teaching(self, message: str) -> tuple[bool, str | None, str | None]:
        message_lower = message.lower().strip()
        pending = self.get_pending_topic()

        if not pending:
            return False, None, None

        # Detect question / topic switch
        is_question = (
            message_lower.endswith('?') or
            message_lower.startswith(('what', 'how', 'why', 'when', 'where', 'which', 'who'))
        )

        line_count = message.count('\n') + 1
        word_count = len(message.split())

        # FIX: If user is providing a detailed answer for pending topic, accept it
        # Check if this is a detailed answer (not a question and has sufficient content)
        if not is_question and (line_count >= 2 or word_count >= 20):
            # Verify it's related to the pending topic
            # Check if pending topic appears in message or message seems like an explanation
            pending_lower = pending.lower()
            if (pending_lower in message_lower or 
                any(word in message_lower for word in ['is', 'are', 'was', 'were', 'has', 'have', 'had']) or
                message_lower.startswith(('it', 'this', 'that', 'the', 'a')) or
                ',' in message or '.' in message):
                return True, pending, None  # Accept as teaching

        if is_question:
            return False, pending, "topic_switch"

        if line_count < 2 or word_count < 20:
            return False, pending, "too_short"

        return True, pending, None


# ============== LLM Service ==============

class LLMService:
    def __init__(self):
        self.model_name = "mock-gpt-4"
        self.max_tokens = 500
        self._learned_context_cache = set()  # Cache of words from learned topics
    
    def update_context_cache(self, learned_topics: dict):
        """ADAPTIVE: Update the context cache with new learned topic keywords"""
        self._learned_context_cache = set()
        for topic in learned_topics.keys():
            self._learned_context_cache.update(topic.lower().split())
        logger.info(f"Updated LLM context cache with {len(self._learned_context_cache)} keywords")
    
    async def extract_intent(self, user_query: str, learned_topics: dict = None) -> dict:
        """
        ADAPTIVE INTENT EXTRACTION: Learns from new contexts automatically
        - search_terms: optimized keywords for RAG search
        - topic: main topic user is asking about
        - intent_type: question/command/info
        - context_hints: what kind of answer user expects
        """
        await asyncio.sleep(0.02)  # Simulated LLM call
        
        # ADAPTIVE: Update context cache if new topics provided
        if learned_topics:
            self.update_context_cache(learned_topics)
        
        query_lower = user_query.lower().strip()
        words = query_lower.replace('?', '').replace(',', '').replace('.', '').split()
        
        # Remove stopwords but keep meaningful words
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 
                     'could', 'would', 'should', 'in', 'to', 'for', 'and', 'or', 'of',
                     'tell', 'me', 'about', 'please', 'give', 'explain', 'describe',
                     'i', 'want', 'know', 'need', 'help', 'with'}
        
        meaningful_words = [w for w in words if w not in stopwords and len(w) > 1]
        
        # ADAPTIVE: Prioritize words that match learned topic keywords
        # This ensures the system adapts to newly learned contexts
        prioritized_words = []
        other_words = []
        for w in meaningful_words:
            if w in self._learned_context_cache:
                prioritized_words.append(w)
            else:
                other_words.append(w)
        
        # Put prioritized words first (they're more likely to match topics)
        search_terms = prioritized_words + other_words
        
        # SMART INTENT EXTRACTION:
        # Identify what KIND of information user wants
        intent_type = "question"
        if any(w in query_lower for w in ['how', 'tutorial', 'guide', 'steps']):
            intent_type = "how_to"
        elif any(w in query_lower for w in ['what is', 'define', 'meaning']):
            intent_type = "definition"
        elif any(w in query_lower for w in ['compare', 'difference', 'vs', 'versus']):
            intent_type = "comparison"
        elif any(w in query_lower for w in ['list', 'types', 'kinds', 'examples']):
            intent_type = "enumeration"
        
        # Extract the MAIN TOPIC - prioritize learned context words
        main_topic = ' '.join(search_terms) if search_terms else query_lower
        
        # Context hints - help RAG understand what to prioritize
        context_hints = {
            "wants_specific": len(meaningful_words) >= 2,
            "has_learned_context": len(prioritized_words) > 0,  # ADAPTIVE hint
            "location_based": any(w in query_lower for w in ['where', 'location', 'place', 'near']),
            "time_based": any(w in query_lower for w in ['when', 'time', 'schedule', 'hours']),
        }
        
        return {
            "original_query": user_query,
            "search_terms": search_terms,
            "topic": main_topic,
            "intent_type": intent_type,
            "context_hints": context_hints,
            "search_query": ' '.join(search_terms)  # Optimized query for RAG
        }
    
    async def generate(self, prompt: str, context: str, is_relevant: bool = True) -> str:
        """Generate response using LLM based on retrieved context"""
        await asyncio.sleep(0.05)  # Reduced latency
        
        # If query is not relevant to knowledge base, return learning message
        if not is_relevant:
            return self._generate_learning_response(prompt)
        
        # Extract the actual content from context (remove relevance scores)
        context_docs = []
        for line in context.split('\n'):
            if line.strip():
                # Extract content after the relevance score
                parts = line.split(') ', 1)
                if len(parts) > 1:
                    context_docs.append(parts[1])
        
        # Build a coherent response from retrieved documents
        question_lower = prompt.lower()
        
        # Find the most relevant document content
        response_parts = []
        
        # Add introduction
        topic = self._identify_topic(question_lower)
        response_parts.append(f"**{topic.title()}**\n")
        
        # Add main content from retrieved documents
        if context_docs:
            response_parts.append(context_docs[0])  # Primary answer
            
            # Add supporting information if available
            if len(context_docs) > 1:
                response_parts.append(f"\n\n**Additional Information:**\n{context_docs[1]}")
            
            if len(context_docs) > 2:
                response_parts.append(f"\n\n**Related Concept:**\n{context_docs[2]}")
        else:
            response_parts.append("I don't have specific information about this topic in my knowledge base.")
        
        return "\n".join(response_parts)
    
    def _generate_learning_response(self, prompt: str) -> str:
        """Generate a response asking user to teach the AI"""
        topic = self._extract_topic_from_question(prompt)
        return f"""**I'm Still Learning! 📚**

I don't have information about **"{topic}"** in my knowledge base yet.

**Please help me learn!** You can teach me by using the Chat section 


Once you teach me, I'll remember it for next time! 🧠"""
    
    def _identify_topic(self, text: str) -> str:
        """Identify the main topic from the question"""
        topic_keywords = {
            "overfitting": "Overfitting",
            "underfitting": "Underfitting", 
            "neural network": "Neural Networks",
            "deep learning": "Deep Learning",
            "machine learning": "Machine Learning",
            "training": "Model Training",
            "gradient descent": "Gradient Descent",
            "backpropagation": "Backpropagation",
            "regularization": "Regularization",
            "dropout": "Dropout",
            "accuracy": "Model Evaluation",
            "precision": "Model Evaluation",
            "recall": "Model Evaluation",
            "loss function": "Loss Functions",
            "loss": "Loss Functions",
            "optimizer": "Optimizers",
            "preprocessing": "Data Preprocessing",
            "feature engineering": "Feature Engineering",
            "transfer learning": "Transfer Learning",
            "cross validation": "Cross Validation",
            "ensemble": "Ensemble Methods",
            "hyperparameter": "Hyperparameter Tuning",
            "supervised": "Supervised Learning",
            "unsupervised": "Unsupervised Learning",
            "activation": "Activation Functions",
            "normalization": "Data Normalization",
            "epoch": "Training Process",
            "batch": "Batch Processing",
            "classification": "Classification",
            "regression": "Regression",
            "clustering": "Clustering",
            "data": "Data Science",
            "model": "Model Architecture",
            "algorithm": "Algorithm",
            "prediction": "Prediction",
            "feature": "Feature Analysis",
            "weight": "Model Weights",
            "bias": "Bias in ML",
            "variance": "Variance",
            "layer": "Neural Network Layers",
            "embedding": "Embeddings",
            "vector": "Vector Representations",
            "cnn": "Convolutional Neural Networks",
            "rnn": "Recurrent Neural Networks",
            "lstm": "LSTM Networks",
            "transformer": "Transformers",
            "attention": "Attention Mechanism",
            "reinforcement": "Reinforcement Learning"
        }
        
        for keyword, topic in topic_keywords.items():
            if keyword in text:
                return topic
        
        # Extract topic from question dynamically
        return self._extract_topic_from_question(text)
    
    def _extract_topic_from_question(self, text: str) -> str:
        """Extract a meaningful topic from the question itself"""
        # Remove common question words
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 
                     'could', 'would', 'should', 'in', 'to', 'for', 'and', 'or', 'of'}
        
        words = text.lower().replace('?', '').replace(',', '').replace('.', '').split()
        meaningful_words = [w.capitalize() for w in words if w not in stopwords and len(w) > 2]
        
        if meaningful_words:
            # Return first 2-3 meaningful words as topic
            topic_words = meaningful_words[:3]
            return ' '.join(topic_words)
        
        return "Your Question"
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract simple keywords from text"""
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 'in', 'to', 'for', 'and', 'or', 'of'}
        words = text.lower().replace('?', '').replace(',', '').split()
        return [w for w in words if w not in stopwords and len(w) > 2]

# ============== Pipeline Orchestrator ==============

class AIPipeline:
    def __init__(self):
        self.vector_store = FAISSVectorStore()
        self.llm = LLMService()
        self.cache: dict[str, PipelineResponse] = {}
    
    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    async def process(self, request: QueryRequest) -> PipelineResponse:
        """Run the complete AI pipeline"""
        start_time = time.perf_counter()
        stages = {}
        
        # Check if pending topic handling is needed
        is_teaching, pending_topic, error_type = self.vector_store.is_user_teaching(request.question)

        if request.question.strip().lower() == "/cancel":
            topic = self.vector_store.get_pending_topic()
            if topic:
                self.vector_store.clear_pending_topic(topic)
                return PipelineResponse(
                    answer=f"❌ Learning for **{topic}** cancelled.",
                    sources=[],
                    latency_ms=0.0,
                    pipeline_stages={"action": "cancelled"}
                )

        if pending_topic and error_type:
            if error_type == "topic_switch":
                return PipelineResponse(
                    answer=(
                        f"⚠️ **Please complete the previous topic first**\n\n"
                        f"You were explaining **'{pending_topic}'**, but you switched to a new question.\n\n"
                        f"👉 Please explain **{pending_topic}** in at least **2 lines**, "
                        f"or use `/cancel` to skip it."
                    ),
                    sources=[],
                    latency_ms=0.0,
                    pipeline_stages={"error": "topic_switched"}
                )

            if error_type == "too_short":
                return PipelineResponse(
                    answer=(
                        f"⚠️ Incomplete explanation please explain the previous topic first otherwise I cannot proceed\n\n"
                        f"Your explanation for '{pending_topic}' is too short.\n\n"
                        f"👉 Please write at least 2 lines (20+ words)."
                    ),
                    sources=[],
                    latency_ms=0.0,
                    pipeline_stages={"error": "answer_too_short"}
                )

        if is_teaching and pending_topic:
            return await self._handle_user_teaching(request.question, pending_topic, start_time)
        
        # Stage 1: Check cache
        cache_key = self._cache_key(request.question)
        if cache_key in self.cache:
            logger.info(f"Cache hit for: {request.question[:50]}...")
            cached = self.cache[cache_key]
            # Return a copy with cache status updated
            return PipelineResponse(
                answer=cached.answer,
                sources=cached.sources,
                latency_ms=0.0,  # Instant from cache
                pipeline_stages={"cache": "hit", "original_latency_ms": cached.latency_ms}
            )
        
        # ========== ADAPTIVE FLOW: LLM → RAG → LLM ==========
        
        # Stage 2: ADAPTIVE LLM - Extract intent using learned context
        stage_start = time.perf_counter()
        # Pass learned_topics so LLM can adapt to new contexts
        intent = await self.llm.extract_intent(
            request.question, 
            learned_topics=self.vector_store.learned_topics
        )
        stages["intent_extraction_ms"] = (time.perf_counter() - stage_start) * 1000
        stages["intent"] = {
            "search_query": intent["search_query"],
            "topic": intent["topic"],
            "intent_type": intent["intent_type"],
            "has_learned_context": intent["context_hints"].get("has_learned_context", False)
        }
        
        # Stage 3: ADAPTIVE RAG Search - Uses intent + adapts to new topics
        stage_start = time.perf_counter()
        search_results = await self.vector_store.search_with_intent(
            intent["search_query"], 
            intent["search_terms"],
            intent["context_hints"],
            request.top_k
        )
        is_relevant = self.vector_store.is_relevant_match(search_results)
        stages["retrieval_ms"] = (time.perf_counter() - stage_start) * 1000
        stages["is_relevant"] = is_relevant
        
        # Stage 4: Context building
        stage_start = time.perf_counter()
        context = self._build_context(search_results) if is_relevant else ""
        sources = [doc[:50] + "..." for doc, _ in search_results] if is_relevant else []
        stages["context_build_ms"] = (time.perf_counter() - stage_start) * 1000
        
        # Stage 5: LLM SECOND - Generate response using original question + RAG context
        stage_start = time.perf_counter()
        answer = await self.llm.generate(request.question, context, is_relevant)
        stages["llm_generation_ms"] = (time.perf_counter() - stage_start) * 1000
        
        # If not relevant, set as pending topic for learning (use LLM-extracted topic)
        if not is_relevant:
            self.vector_store.set_pending_topic(intent["topic"])
        
        # Stage 6: Post-processing
        stage_start = time.perf_counter()
        final_answer = self._postprocess_response(answer)
        stages["postprocessing_ms"] = (time.perf_counter() - stage_start) * 1000
        
        total_latency = (time.perf_counter() - start_time) * 1000
        stages["cache"] = "miss"
        
        response = PipelineResponse(
            answer=final_answer,
            sources=sources,
            latency_ms=round(total_latency, 2),
            pipeline_stages=stages
        )
        
        # Cache the response
        self.cache[cache_key] = response
        
        # Auto-add user query to FAISS vector database
        self.vector_store.add_user_query(request.question)
        
        logger.info(f"Pipeline completed in {total_latency:.2f}ms")
        
        return response
    
    async def _handle_user_teaching(self, user_message: str, topic: str, start_time: float) -> PipelineResponse:
        """Handle when user is teaching us about a topic"""
        # Learn the topic from user's message
        self.vector_store.learn_topic(topic, user_message, self.cache)
        
        # Clear the pending topic
        self.vector_store.clear_pending_topic(topic)
        
        # Clear any related cache
        topic_cache_key = self._cache_key(topic)
        if topic_cache_key in self.cache:
            del self.cache[topic_cache_key]
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        thank_you_message = f"""**Thank You for Sharing Your Insight! 🙏**

I've learned about **"{topic.title()}"** from your explanation:

> {user_message[:200]}{'...' if len(user_message) > 200 else ''}

**This knowledge is now stored in my database!** 📚

Next time you or anyone asks about "{topic}", I'll be able to help! Feel free to ask me about it now. 🧠"""
        
        response = PipelineResponse(
            answer=thank_you_message,
            sources=[f"Learned from user: {topic}"],
            latency_ms=round(total_latency, 2),
            pipeline_stages={"action": "learned_from_user", "topic": topic}
        )
        
        logger.info(f"Learned topic '{topic}' from user input")
        
        return response
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the query"""
        return query.strip().lower()
    
    def _build_context(self, results: list[tuple[str, float]]) -> str:
        """Build context string from search results"""
        context_parts = []
        for i, (content, score) in enumerate(results, 1):
            context_parts.append(f"[{i}] (relevance: {score:.2f}) {content}")
        return "\n".join(context_parts)
    
    def _postprocess_response(self, response: str) -> str:
        """Clean up the LLM response"""
        return response.strip()

# ============== Initialize Pipeline ==============

pipeline = AIPipeline()

# ============== API Endpoints ==============

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    faiss_stats = pipeline.vector_store.get_stats()
    return {
        "status": "healthy",
        "model": pipeline.llm.model_name,
        "docs_indexed": len(pipeline.vector_store.entries),
        "cache_size": len(pipeline.cache),
        "faiss_stats": faiss_stats
    }

@app.post("/query", response_model=PipelineResponse)
async def query_pipeline(request: QueryRequest):
    """Main query endpoint for the AI pipeline"""
    try:
        response = await pipeline.process(request)
        return response
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_document(doc: Document):
    """Add a document to the knowledge base"""
    embedding = pipeline.vector_store._compute_embedding(doc.content)
    pipeline.vector_store.add_to_index(embedding)
    entry = VectorEntry(doc.id, doc.content, embedding.tolist(), doc.metadata)
    pipeline.vector_store.entries.append(entry)
    return {"status": "added", "doc_id": doc.id, "faiss_total": pipeline.vector_store.index.ntotal}

@app.get("/user-queries")
async def list_user_queries():
    """List all user queries stored in FAISS"""
    return {
        "count": len(pipeline.vector_store.user_queries),
        "queries": pipeline.vector_store.user_queries
    }

@app.get("/faiss-stats")
async def faiss_stats():
    """Get FAISS index statistics"""
    return pipeline.vector_store.get_stats()

class LearnRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=2000)

@app.post("/learn")
async def learn_topic(request: LearnRequest):
    """Teach the AI a new topic"""
    try:
        topic_lower = request.topic.lower().strip()
        is_update = topic_lower in pipeline.vector_store.learned_topics
        
        # Pass cache reference to clear related entries
        pipeline.vector_store.learn_topic(request.topic, request.description, pipeline.cache)
        
        # Also clear any cache entry that matches this topic query
        topic_cache_key = pipeline._cache_key(request.topic)
        if topic_cache_key in pipeline.cache:
            del pipeline.cache[topic_cache_key]
            logger.info(f"Cleared direct cache for topic: {request.topic}")
        
        action = "updated" if is_update else "learned"
        
        # This is the message the user wants
        thank_you_message = f"""**Thank You for Sharing Your Insight! 🙏**

I've learned about **"{request.topic.title()}"** from your explanation:

> {request.description[:200]}{'...' if len(request.description) > 200 else ''}

**This knowledge is now stored in my database!** 📚

Next time you or anyone asks about "{request.topic}", I'll be able to help! Feel free to ask me about it now. 🧠"""
        
        return {
            "status": action,
            "topic": request.topic,
            "answer": thank_you_message,
            "total_learned_topics": len(pipeline.vector_store.learned_topics)
        }
    except Exception as e:
        logger.error(f"Learn error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learned-topics")
async def list_learned_topics():
    """List all topics the AI has learned"""
    return {
        "count": len(pipeline.vector_store.learned_topics),
        "topics": pipeline.vector_store.learned_topics
    }

@app.delete("/learned-topics/{topic}")
async def delete_learned_topic(topic: str):
    """Delete a specific learned topic - AI will learn it fresh next time"""
    # URL decode the topic
    import urllib.parse
    topic_decoded = urllib.parse.unquote(topic)
    
    success = pipeline.vector_store.delete_topic(topic_decoded)
    
    if success:
        # Clear cache to remove old responses
        pipeline.cache.clear()
        return {
            "status": "deleted",
            "topic": topic_decoded,
            "message": "Topic deleted. AI will learn it fresh when asked again."
        }
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Topic '{topic_decoded}' not found in learned topics"
        )

@app.delete("/learned-topics")
async def delete_all_learned_topics():
    """Delete ALL learned topics - AI will start fresh"""
    count = len(pipeline.vector_store.learned_topics)
    pipeline.vector_store.learned_topics.clear()
    pipeline.vector_store._validate_and_sync_data()
    pipeline.vector_store._initialize_knowledge_base()
    pipeline.vector_store._save_persisted_data()
    pipeline.cache.clear()
    
    return {
        "status": "cleared",
        "topics_removed": count,
        "message": "All learned topics deleted. AI will learn fresh when asked."
    }

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    return {
        "count": len(pipeline.vector_store.entries),
        "documents": [
            {"id": e.doc_id, "preview": e.content[:100]}
            for e in pipeline.vector_store.entries
        ]
    }

@app.delete("/cache")
async def clear_cache():
    """Clear the response cache"""
    count = len(pipeline.cache)
    pipeline.cache.clear()
    return {"status": "cleared", "entries_removed": count}



# ============== Run Server ==============


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("Starting AI Pipeline Server")
    print("="*60)
    print(f"URL: http://127.0.0.1:60922")
    print(f"Health Check: http://127.0.0.1:60922/health")
    print("="*60)
    uvicorn.run(app, host="127.0.0.1", port=60922, log_level="info")