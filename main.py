import sys
from preprocessing import main as preprocess_main
from indexing import build_inverted_index
from ranking import load_query, rank_documents
import json

documents = preprocess_main(corpus_path="scifact/corpus.jsonl")
inverted_index = build_inverted_index(documents)

with open("scifact/queries.jsonl", "r") as f:
    queries = [json.loads(line) for line in f]

# Filter only odd-numbered queries (test queries)
test_queries = [q for q in queries if int(q["_id"]) % 2 == 1]

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

with open("Results", "w") as file:
    for query in test_queries:
        ranked = rank_documents(query["text"], inverted_index)

        for i, (doc_id, score) in enumerate(ranked[:100], 1):
            file.write(f"{query['_id']} Q0 {doc_id} {i} {score:.4f} my_system\n")
