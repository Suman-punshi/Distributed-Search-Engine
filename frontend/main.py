# frontend/main.py

from flask import Flask, request, render_template
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

EXTRACTOR_URL = "http://extractor:5001/extract"
SEARCHER_URL = "http://searcher:5002/search"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    file = request.files['image']
    if not file:
        return "No image uploaded", 400

    response = requests.post(EXTRACTOR_URL, files={"image": file})
    if response.status_code != 200:
        return "Feature extraction failed", 500

    vector = response.json()["vector"]
    search_response = requests.post(SEARCHER_URL, json={"vector": vector})

    if search_response.status_code != 200:
        return "Search failed", 500

    results = search_response.json()["results"]
    return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
