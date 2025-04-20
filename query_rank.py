import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os
import tensorflow_hub as hub
import requests
from io import BytesIO
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone

# ========== CONFIG ========== 
BOOK_CSV = 'main_dataset.csv'
IMG_BASE_PATH = '.'  # Base path for your book images
MODULE_URL = "https://tfhub.dev/google/delf/1"
PINECONE_API_KEY = "pcsk_zhkLX_3ofVYvUgWWFeGhEjdftpy3wWj3wCbv6EmYhm2EEKNMVft31uycr1ZfR31LsJdV2"  # Replace with your Pinecone API key
PINECONE_ENV = "us-west1-gcp"
INDEX_NAME = "book-search-engine"

# ========== PINECONE INIT ==========
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ========== DELF MODEL ==========
print("Loading DELF model from TF Hub...")
delf_model = hub.load(MODULE_URL).signatures['default']


# ========== FEATURE EXTRACTOR ========== 
def extract_feature(image):
    try:
        img = image.convert('RGB')
        img = ImageOps.fit(img, (320, 320), Image.LANCZOS)
        img = img.resize((320, 320))
        np_image = np.array(img)
        float_image = tf.image.convert_image_dtype(np_image, tf.float32)
        
        result = delf_model(
            image=float_image,
            score_threshold=tf.constant(50.0),
            image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            max_feature_num=tf.constant(1000)
        )
        
        features = result['descriptors'].numpy()

        # Edge case: No descriptors returned
        if features.size == 0:
            print(f"No descriptors found in image.")
            return None

        pooled = np.mean(features, axis=0)

        # Handle NaNs before normalizing
        if np.isnan(pooled).any():
            print(f"NaN values found in pooled descriptor.")
            return None

        pooled = normalize(pooled.reshape(1, -1)).astype('float32')
        return pooled

    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None


def query_image(image_path_or_url):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path_or_url)

    query_feature = extract_feature(img)
    if query_feature is None:
        print("Query image feature extraction failed.")
        return []

    # Ensure vector is a flat list of native Python floats
    query_feature = query_feature.flatten()
    query_feature = [float(x) for x in query_feature]  # This ensures valid type

    # Query Pinecone (correct API call for pinecone>=3.x)
    query_result = index.query(
        vector=query_feature,
        top_k=10,
        include_metadata=True
        # include_values=False if you don’t need full vectors back
    )

    # Parse results
    top_books = []
    for match in query_result.matches:
        top_books.append({
            "book_id": match.id,
            "similarity": match.score,
            "meta": match.metadata
        })

    return top_books


# Example usage with a local image:
query_local_image = "./0000001.jpg"  # Replace with your local image file path
top_books_local = query_image(query_local_image)

print("\nTop Ranked Books from Local Image Query:")
for book in top_books_local:
    print(f"Book ID: {book['book_id']}, Similarity: {book['similarity']}")
    print(f"Book Meta: {book['meta']}")