# Searcher/searcher.py

from flask import Flask, request, jsonify
from pinecone import Pinecone
import time

from logger_setup import setup_logger
log = setup_logger('searcher', 'logs/searcher.log')

PINECONE_API_KEY = "pcsk_zhkLX_3ofVYvUgWWFeGhEjdftpy3wWj3wCbv6EmYhm2EEKNMVft31uycr1ZfR31LsJdV2"
INDEX_NAME = "book-search-engine"

app = Flask(__name__)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

@app.route("/search", methods=["POST"])
def search():
    try:
        t = time.time()
        vector = request.get_json().get("vector", [])
        results = index.query(vector=vector, top_k=10, include_metadata=True)
        log.info(f"[Pinecone Query] took {time.time() - t:.4f}s")
        return jsonify({
            "results": [
                {"book_id": m.id, "similarity": m.score, "meta": m.metadata}
                for m in results.matches
            ]
        })
    except Exception as e:
        log.error(f"Error during Pinecone query: {e}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
