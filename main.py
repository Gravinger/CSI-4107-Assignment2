import sys
import json
from preprocessing import main as preprocess_main
from indexing import build_inverted_index
from ranking import rank_documents
from neural_reranker import (
    biencoder_rerank,
    crossencoder_rerank,
    precompute_doc_embeddings,
)

# Load and preprocess corpus
doc_dict = preprocess_main(corpus_path="scifact/corpus.jsonl")

# Convert preprocessed dict to list format expected by build_inverted_index
documents_for_index = []
for doc_id, doc_data in doc_dict.items():
    tokens = doc_data.get("title", "").split() + doc_data.get("text", "").split()
    documents_for_index.append({"id": doc_id, "tokens": tokens})

inverted_index = build_inverted_index(documents_for_index)

# Load raw corpus text for neural re-ranking
raw_docs = {}
with open("scifact/corpus.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        raw_docs[d["_id"]] = d["title"] + " " + d["text"]

# Load and filter test queries
with open("scifact/queries.jsonl", "r") as f:
    queries = [json.loads(line) for line in f]

# Filter only queries that appear in test.tsv
test_query_ids = set()
with open("scifact/qrels/test.tsv", "r") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        test_query_ids.add(parts[0])

# Filter only queries that appear in test.tsv
test_queries = [q for q in queries if str(q["_id"]) in test_query_ids]

# Sort ascending
test_queries = sorted(test_queries, key=lambda x: int(x["_id"]))

# Pre-encode all documents once
precompute_doc_embeddings(raw_docs)

# Retrieve top-100 candidates with TF-IDF, then re-rank with bi-encoder
with open("Results_biencoder", "w") as file:
    for query in test_queries:
        candidates = rank_documents(query["text"], inverted_index)[:100]
        reranked = biencoder_rerank(query["text"], candidates, raw_docs)
        for i, (doc_id, score) in enumerate(reranked, 1):
            file.write(f"{query['_id']} Q0 {doc_id} {i} {score:.4f} biencoder\n")

# Retrieve top-100 candidates with TF-IDF, then re-rank with cross-encoder
with open("Results_crossencoder", "w") as file:
    for query in test_queries:
        candidates = rank_documents(query["text"], inverted_index)[:100]
        reranked = crossencoder_rerank(query["text"], candidates, raw_docs)
        for i, (doc_id, score) in enumerate(reranked, 1):
            file.write(f"{query['_id']} Q0 {doc_id} {i} {score:.4f} crossencoder\n")