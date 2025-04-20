import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
from pinecone import Pinecone
from tqdm import tqdm

# ========== CONFIG ==========
FEATURE_DIR = 'features'
BOOK_CSV = 'main_dataset.csv'
IMG_BASE_PATH = '.'
MODULE_URL = "https://tfhub.dev/google/delf/1"
PINECONE_API_KEY = "pcsk_zhkLX_3ofVYvUgWWFeGhEjdftpy3wWj3wCbv6EmYhm2EEKNMVft31uycr1ZfR31LsJdV2"  # Replace with your Pinecone API key
PINECONE_ENV = "us-west1-gcp"      # Replace with your Pinecone environment
INDEX_NAME = "book-search-engine"
# ============================

os.makedirs(FEATURE_DIR, exist_ok=True)

# Load DELF model
print("Loading DELF model from TF Hub...")
delf_model = hub.load(MODULE_URL).signatures['default']

# Load book metadata
df = pd.read_csv(BOOK_CSV)
df = df.reset_index().rename(columns={"index": "book_id"})
df.to_json('book_metadata.json', orient='records', lines=True)
id_to_meta = {row['book_id']: row for _, row in df.iterrows()}

# ========== FEATURE EXTRACTOR ==========
def extract_feature(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
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
            print(f"No descriptors found in {image_path}")
            return None

        pooled = np.mean(features, axis=0)

        # Handle NaNs before normalizing
        if np.isnan(pooled).any():
            print(f"NaN values found in pooled descriptor of {image_path}")
            return None

        pooled = normalize(pooled.reshape(1, -1)).astype('float32')
        return pooled

    except Exception as e:
        print(f"Error extracting feature from {image_path}: {e}")
        return None

    

# ========== STEP 1: Extract Features ==========
print("Extracting and saving features...")
features = []
book_ids = []

for i, row in tqdm(df.iterrows()):
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

# ========== STEP 2: Upload to Pinecone ==========
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print("Uploading vectors to Pinecone...")
vectors_to_upsert = []
for i, vec in enumerate(features):
    meta = id_to_meta.get(int(book_ids[i]), {})
    meta_dict = meta.to_dict() if isinstance(meta, pd.Series) else meta
    pinecone_id = str(book_ids[i])
    print(vec.shape)         # Should be (128,)
    print(len(vec.tolist())) # Should be 128
    print(f"Upserting: ID={pinecone_id}, Vector Dim={len(vec)}, Meta Keys={list(meta_dict.keys())}")

    vectors_to_upsert.append((pinecone_id, vec.tolist(), meta_dict))

batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i+batch_size]
    index.upsert(vectors=batch)

print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone.")


response = index.fetch(ids=["0"])
print(response)
