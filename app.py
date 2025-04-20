from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os
import tensorflow_hub as hub
import requests
from io import BytesIO
from sklearn.preprocessing import normalize
from pinecone import Pinecone
import cv2
import matplotlib.pyplot as plt


# Flask app initialization
app = Flask(__name__)

# ========== CONFIG ========== 
MODULE_URL = "https://tfhub.dev/google/delf/1"
PINECONE_API_KEY = "pcsk_zhkLX_3ofVYvUgWWFeGhEjdftpy3wWj3wCbv6EmYhm2EEKNMVft31uycr1ZfR31LsJdV2"  
INDEX_NAME = "book-search-engine"

# ========== PINECONE INIT ==========
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ========== DELF MODEL ==========
print("Loading DELF model from TF Hub...")
delf_model = hub.load(MODULE_URL).signatures['default']



def detect_cover(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:  # ignore tiny boxes
            cropped = open_cv_image[y:y+h, x:x+w]

            # Show cropped image
            plt.imshow(cropped)
            plt.axis('off')
            plt.title("Detected Book Cover")
            plt.show()

            return Image.fromarray(cropped)

    # Show original if no crop found
    plt.imshow(open_cv_image)
    plt.axis('off')
    plt.title("Fallback: Original Image")
    plt.show()

    return image


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

        if features.size == 0:
            print(f"No descriptors found in image.")
            return None

        pooled = np.mean(features, axis=0)

        if np.isnan(pooled).any():
            print(f"NaN values found in pooled descriptor.")
            return None

        pooled = normalize(pooled.reshape(1, -1)).astype('float32')
        return pooled

    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None


# ========== ROUTES ========== 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return "No image uploaded"
    image_file = request.files['image']

    if image_file.filename == '':
        return "No selected file"

    img = Image.open(image_file.stream)
    img = detect_cover(img)
    query_feature = extract_feature(img)
    if query_feature is None:
        return "Feature extraction failed"

    query_feature = query_feature.flatten()
    query_feature = [float(x) for x in query_feature]

    query_result = index.query(
        vector=query_feature,
        top_k=10,
        include_metadata=True
    )

    top_books = []
    for match in query_result.matches:
        top_books.append({
            "book_id": match.id,
            "similarity": match.score,
            "meta": match.metadata
        })

    return render_template('results.html', results=top_books)

# ========== MAIN ========== 
if __name__ == '__main__':
    app.run(debug=True)
