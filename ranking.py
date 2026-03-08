import json
import math
from collections import Counter

from preprocessing import read_stopwords, preprocess_query


def get_document_count(inverted_index):
    doc_ids = set()
    for postings in inverted_index.values():
        doc_ids.update(postings.keys())
    return len(doc_ids)


def compute_idf_and_doc_lengths(inverted_index, N):
    idf = {}
    doc_length_sq = {}

    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log(N / df) if df > 0 else 0.0

        for doc_id, tf in postings.items():
            weight = (1 + math.log(tf)) * idf[term] if tf > 0 else 0.0
            doc_length_sq[doc_id] = doc_length_sq.get(doc_id, 0.0) + weight * weight

    doc_lengths = {doc_id: math.sqrt(sq) for doc_id, sq in doc_length_sq.items()}
    return idf, doc_lengths


def get_candidate_docs(query_tokens, inverted_index):
    candidates = set()
    for token in query_tokens:
        if token in inverted_index:
            candidates.update(inverted_index[token].keys())
    return candidates


def cosine_similarity(query_tokens, doc_id, inverted_index, idf, doc_lengths):
    q_tf = Counter(query_tokens)
    q_weights = {}
    q_norm_sq = 0.0
    for term, tf in q_tf.items():
        if term not in idf:
            continue
        w = (1 + math.log(tf)) * idf[term] if tf > 0 else 0.0
        q_weights[term] = w
        q_norm_sq += w * w
    q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

    dot = 0.0
    for term, q_w in q_weights.items():
        if term not in inverted_index or doc_id not in inverted_index[term]:
            continue
        tf_d = inverted_index[term][doc_id]
        d_w = (1 + math.log(tf_d)) * idf[term] if tf_d > 0 else 0.0
        dot += q_w * d_w

    d_norm = doc_lengths.get(doc_id, 0.0)
    if d_norm <= 0:
        return 0.0
    return dot / (q_norm * d_norm)


def rank_documents(query_text, inverted_index):
    N = get_document_count(inverted_index)
    if N == 0:
        return []

    stopwords = read_stopwords()
    query_tokens = preprocess_query(query_text, stopwords)
    if not query_tokens:
        return []

    idf, doc_lengths = compute_idf_and_doc_lengths(inverted_index, N)
    candidates = get_candidate_docs(query_tokens, inverted_index)

    scores = []
    for doc_id in candidates:
        sim = cosine_similarity(query_tokens, doc_id, inverted_index, idf, doc_lengths)
        scores.append((doc_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def load_query(queries_path, query_id):
    with open(queries_path, "r") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) == str(query_id):
                return q
    return None
