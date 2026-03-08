import argparse
import json
import re
from porter_stemmer import PorterStemmer

STOPWORDS_PATH = "./stopwords.txt"
SPACE_PUNCTUATION = re.compile(r"[:\\/\-]")
PUNCTUATION = re.compile(r"[\.:;\\,\(\)\[\]\{\}\?!@\^\$\*&#%\-\+=_/<>`~\'\"0-9]")


def binary_search(word, stopwords) -> bool:
    start = 0
    end = len(stopwords) - 1

    while start <= end:
        index = (start + end) // 2

        if stopwords[index].lower() == word:
            return True

        elif word < stopwords[index].lower():
            end = index - 1

        else:
            start = index + 1

    return False


def preprocess_docs(documents, stopwords):

    stemmer = PorterStemmer()
    preprocessed_documents = []

    for doc in documents:
        new_doc = {"id": doc["_id"], "tokens": []}

        title = re.sub(SPACE_PUNCTUATION, " ", doc["title"])

        for word in title.split(" "):
            word = re.sub(PUNCTUATION, "", word)
            word = word.lower().strip()

            if word == "":
                continue

            if not binary_search(word, stopwords):
                word = stemmer.stem(word, 0, len(word) - 1)

                new_doc["tokens"].append(word)

        text = re.sub(SPACE_PUNCTUATION, " ", doc["text"])

        for word in text.split(" "):
            word = re.sub(PUNCTUATION, "", word)
            word = word.lower().strip()

            if word == "":
                continue

            if not binary_search(word, stopwords):
                word = stemmer.stem(word, 0, len(word) - 1)

                new_doc["tokens"].append(word)

        preprocessed_documents.append(new_doc)

    return preprocessed_documents


def preprocess_query(text, stopwords):
    stemmer = PorterStemmer()
    tokens = []

    text = re.sub(SPACE_PUNCTUATION, " ", text)

    for word in text.split(" "):
        word = re.sub(PUNCTUATION, "", word)
        word = word.lower().strip()
        if word and not binary_search(word, stopwords):
            word = stemmer.stem(word, 0, len(word) - 1)
            tokens.append(word)
    return tokens


def read_stopwords():
    stopwords = []

    # Try to open the stopwords document
    try:
        with open(STOPWORDS_PATH, "r") as stopwords_file:
            for line in stopwords_file:
                # Save the stopword to the result list
                stopwords.append(line.strip())

    except Exception as e:
        print("Failed to open stopwords document due to error:", e)

    return stopwords


def main(corpus_path):
    stopwords_list = read_stopwords()

    raw_documents = []

    try:
        with open(corpus_path, "r") as corpus_file:
            # For each line in the corpus, try to convert the line to a dictionary
            for line in corpus_file:
                try:
                    raw_documents.append(json.loads(line))

                except Exception as e:
                    print("A line is in the wrong format and failed to convert to a dictionary due to error: ", e)

    except Exception as e:
        print("Failed to open corpus document due to error:", e)

    if len(raw_documents) <= 0:
        print("No valid corpus documents were found. Ending program.")
        return []
    else:
        # Get the result of the preprocessing on the documents
        result = preprocess_docs(raw_documents, stopwords_list)
        return result
    # print(result[len(result) - 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", metavar="path", required=True, help="The path to the corpus file")

    main(corpus_path=parser.parse_args().corpus_path)
