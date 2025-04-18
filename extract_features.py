import os
import faiss
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import normalize
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import cv2
import json

# ========== CONFIG ==========
NUM_SHARDS = 4
FEATURE_DIR = 'features'      # Folder to save .npy features
INDEX_DIR = 'faiss_indexes'   # Folder to save FAISS indexes
BOOK_CSV = 'books.csv'        # Your full books.csv file
IMG_BASE_PATH = '.'           # Root dir for book images
MODULE_URL = "https://tfhub.dev/google/delf/1"
# ============================

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load DELF model
print("Loading DELF model from TF Hub...")
delf_model = hub.load(MODULE_URL).signatures['default']

# Load book metadata
df = pd.read_csv(BOOK_CSV)
df = df.reset_index().rename(columns={"index": "book_id"})  # Add book_id column

# Save metadata for later lookup
df.to_json('book_metadata.json', orient='records', lines=True)
id_to_meta = {row['book_id']: row for _, row in df.iterrows()}

# ========== FEATURE EXTRACTOR ==========
def extract_feature(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((320, 320))  # Resize for consistency
        img_np = np.array(img) / 255.0
        img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        result = delf_model(img_tensor)
        features = result['descriptors'].numpy()

        # Use mean-pooled descriptor as image vector (simplified)
        pooled = np.mean(features, axis=0)
        pooled = normalize(pooled.reshape(1, -1)).astype('float32')
        return pooled
    except Exception as e:
        print(f"Error extracting feature from {image_path}: {e}")
        return None
# ========================================

# ========== STEP 1: Extract Features ==========
print("Extracting and saving features...")
features = []
book_ids = []

for i, row in df.iterrows():
    img_path = row['img_paths']
    full_img_path = os.path.join(IMG_BASE_PATH, img_path)
    vec = extract_feature(full_img_path)
    if vec is not None:
        features.append(vec[0])
        book_ids.append(row['book_id'])

features = np.array(features).astype('float32')
book_ids = np.array(book_ids)

np.save(os.path.join(FEATURE_DIR, 'features.npy'), features)
np.save(os.path.join(FEATURE_DIR, 'book_ids.npy'), book_ids)

# ========== STEP 2: Create Pseudo-Distributed Index ==========
print("Creating FAISS shards...")
features = np.load(os.path.join(FEATURE_DIR, 'features.npy'))
book_ids = np.load(os.path.join(FEATURE_DIR, 'book_ids.npy'))

shard_size = len(features) // NUM_SHARDS
for i in range(NUM_SHARDS):
    start = i * shard_size
    end = None if i == NUM_SHARDS - 1 else (i + 1) * shard_size
    shard_feats = features[start:end]
    shard_ids = book_ids[start:end]

    index = faiss.IndexFlatL2(shard_feats.shape[1])
    index.add(shard_feats)

    faiss.write_index(index, os.path.join(INDEX_DIR, f'shard_{i}.index'))
    np.save(os.path.join(INDEX_DIR, f'shard_{i}_ids.npy'), shard_ids)

print(f"Indexed {len(features)} book covers into {NUM_SHARDS} FAISS shards.")
