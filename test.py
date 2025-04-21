import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

FRONTEND_URL = "http://localhost:5000/search"  # or the actual IP if deployed
TEST_IMAGE_DIR = "./test_images"
CONCURRENT_USERS = 10

def send_image(file_path):
    try:
        with open(file_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(FRONTEND_URL, files=files, timeout=10)
            return (file_path, response.status_code, len(response.content))
    except Exception as e:
        return (file_path, "ERROR", str(e))

def run_load_test():
    image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.jpg')]
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [executor.submit(send_image, img) for img in image_files]
        for future in as_completed(futures):
            results.append(future.result())

    for res in results:
        print(f"{res[0]} -> Status: {res[1]}, Length: {res[2]}")

if __name__ == "__main__":
    run_load_test()
