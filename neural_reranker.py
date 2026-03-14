"""
Neural re-ranking methods using sentence-transformers.

Method 1 (bi-encoder):
  - We load a pre-trained model (all-MiniLM-L6-v2) that converts text into dense vectors.
  - Before processing queries, encode all documents once and cache the results.
  - For each query, we encode it into a vector and re-rank TF-IDF candidates by cosine similarity.

Method 2 (cross-encoder):
  - We load a pre-trained model (cross-encoder/ms-marco-MiniLM-L-6-v2).
  - For each (query, candidate-document) pair, the model predicts a relevance score directly.
  - We sort candidates by that score.

INSTALL DEPENDENCY:
  pip install sentence-transformers
"""

from sentence_transformers import CrossEncoder, SentenceTransformer, util

# -------------------- Bi-encoder setup --------------------
BIENCODER_MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading bi-encoder model: {BIENCODER_MODEL_NAME} ...")
_biencoder_model = SentenceTransformer(BIENCODER_MODEL_NAME)
print("Bi-encoder model loaded.")

# -------------------- Cross-encoder setup --------------------
CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_crossencoder_model = None

# Cache for pre-encoded document embeddings
_doc_embeddings_cache = {}


def _get_crossencoder_model():
    global _crossencoder_model
    if _crossencoder_model is None:
        print(f"Loading cross-encoder model: {CROSSENCODER_MODEL_NAME} ...")
        _crossencoder_model = CrossEncoder(CROSSENCODER_MODEL_NAME)
        print("Cross-encoder model loaded.")
    return _crossencoder_model

# Pre-encode all documents once and store in cache, call this before running any queries
def precompute_doc_embeddings(raw_docs):
    global _doc_embeddings_cache
    print(f"Pre-encoding {len(raw_docs)} documents (runs once)...")
    doc_ids = list(raw_docs.keys())
    doc_texts = [raw_docs[doc_id] for doc_id in doc_ids]

    # Encode all documents in one batch
    embeddings = _biencoder_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    for i, doc_id in enumerate(doc_ids):
        _doc_embeddings_cache[doc_id] = embeddings[i]
    print("Document encoding complete.")

# Re-rank a list of candidate documents using a bi-encoder
def biencoder_rerank(query_text, candidates, raw_docs, top_k=100):

    if not candidates:
        return []

    # Encode the query into a dense vector
    query_embedding = _biencoder_model.encode(query_text, convert_to_tensor=True)

    # Score each candidate using cached document embeddings
    scored = []
    for doc_id, _ in candidates:
        if doc_id in _doc_embeddings_cache:
            score = float(util.cos_sim(query_embedding, _doc_embeddings_cache[doc_id]))
            scored.append((doc_id, score))

    # Sort by cosine similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# Re-rank a list of candidate documents using a cross-encoder
def crossencoder_rerank(query_text, candidates, raw_docs, top_k=100, batch_size=32):

    if not candidates:
        return []

    model = _get_crossencoder_model()

    doc_ids = []
    pairs = []

    for doc_id, _ in candidates:
        doc_text = raw_docs.get(doc_id)
        if doc_text is None:
            continue
        doc_ids.append(doc_id)
        pairs.append([query_text, doc_text])

    if not pairs:
        return []

    scores = model.predict(pairs, batch_size=batch_size)
    scored = [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids))]

    # Sort by predicted relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
