"""
Bi-encoder re-ranking using sentence-transformers.

We load a pre-trained model (all-MiniLM-L6-v2) that converts text into dense vectors.
Before processing queries, encode all documents once and cache the results.
For each query, we encode it into a vector and re-rank the TF-IDF candidates
by cosine similarity between the query and each candidate document vector.

INSTALL DEPENDENCY:
  pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util

# Load model once at import time so it isn't reloaded for every query
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading sentence-transformer model: {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# Cache for pre-encoded document embeddings
_doc_embeddings_cache = {}

# Pre-encode all documents once and store in cache, call this before running any queries
def precompute_doc_embeddings(raw_docs):
    global _doc_embeddings_cache
    print(f"Pre-encoding {len(raw_docs)} documents (runs once)...")
    doc_ids = list(raw_docs.keys())
    doc_texts = [raw_docs[doc_id] for doc_id in doc_ids]

    # Encode all documents in one batch
    embeddings = _model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    for i, doc_id in enumerate(doc_ids):
        _doc_embeddings_cache[doc_id] = embeddings[i]
    print("Document encoding complete.")

# Re-rank a list of candidate documents using a bi-encoder
def biencoder_rerank(query_text, candidates, raw_docs, top_k=100):

    if not candidates:
        return []

    # Encode the query into a dense vector
    query_embedding = _model.encode(query_text, convert_to_tensor=True)

    # Score each candidate using cached document embeddings
    scored = []
    for doc_id, _ in candidates:
        if doc_id in _doc_embeddings_cache:
            score = float(util.cos_sim(query_embedding, _doc_embeddings_cache[doc_id]))
            scored.append((doc_id, score))

    # Sort by cosine similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
