# Searcher/searcher.py

from flask import Flask, request, jsonify
from pinecone import Pinecone
import os

PINECONE_API_KEY = "pcsk_zhkLX_3ofVYvUgWWFeGhEjdftpy3wWj3wCbv6EmYhm2EEKNMVft31uycr1ZfR31LsJdV2"
INDEX_NAME = "book-search-engine"

app = Flask(__name__)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    vector = data.get("vector", [])
    if not vector:
        return jsonify({"error": "No vector provided"}), 400

    query_result = index.query(vector=vector, top_k=10, include_metadata=True)

    results = [
        {
            "book_id": match.id,
            "similarity": match.score,
            "meta": match.metadata
        }
        for match in query_result.matches
    ]

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
