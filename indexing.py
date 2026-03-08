def build_inverted_index(documents):
    """
    Build an inverted index from preprocessed documents.

    Each token maps to the documents it appears in, with counts.

    Parameters:
        documents (list): List of dicts with id and tokens keys.

    Returns:
        dict: Inverted index { token: { doc_id: count, ... }, ... }
    """
    inverted_index = {}  # Initialize empty dictionary for the index

    for doc in documents:
        doc_id = doc['id']            # Get document ID
        tokens = doc['tokens']        # Get list of tokens for this document

        for token in tokens:
            # If token is not yet in the index, add it
            if token not in inverted_index:
                inverted_index[token] = {}

            # If this document hasn't been counted for this token yet, initialize count
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0

            # Increment the count of this token in this document
            inverted_index[token][doc_id] += 1

    return inverted_index
