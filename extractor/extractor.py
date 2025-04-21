# extractor

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from sklearn.preprocessing import normalize
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os

# Config
MODULE_PATH = "./models/delf/1"
app = Flask(__name__)
delf_model = hub.load(MODULE_PATH).signatures['default']

def detect_cover(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            cropped = open_cv_image[y:y+h, x:x+w]
            return Image.fromarray(cropped)

    return image

@app.route("/extract", methods=["POST"])
def extract_feature():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files['image'].stream).convert("RGB")
    image = detect_cover(image)


    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)

    result = delf_model(
        image=float_image,
        score_threshold=tf.constant(50.0),
        image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
        max_feature_num=tf.constant(1000)
    )

    features = result['descriptors'].numpy()
    if features.size == 0:
        return jsonify({"error": "No descriptors found"}), 500

    pooled = np.mean(features, axis=0)
    pooled = normalize(pooled.reshape(1, -1)).astype('float32')

    return jsonify({"vector": pooled.flatten().tolist()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
