import os
import requests
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import io
import time
import json

# --- Configuration ---
DATABRICKS_HOST = "https://your-databricks-workspace-url.cloud.databricks.com"
DATABRICKS_TOKEN = "dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
SERVING_ENDPOINT_NAME = "resnet50-image-serving-endpoint"  # Use your deployed endpoint name
SERVING_ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"
IMAGE_FOLDER = "./images_to_process"
MAX_WORKERS = 8
BATCH_SIZE = 4  # Consider increasing this for efficiency

# --- Helper Functions ---
def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Encoding error ({image_path}): {e}")
        return None

def send_inference_request(images_base64_batch, image_paths_batch):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps({
        "dataframe_split": {
            "columns": ["image_data"],
            "data": [[img_str] for img_str in images_base64_batch]
        }
    })

    try:
        response = requests.post(SERVING_ENDPOINT_URL, headers=headers, data=data)
        response.raise_for_status()
        return response.json(), image_paths_batch
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response else 'N/A'
        text = e.response.text if e.response else 'No response text'
        print(f"Request error ({status}): {text}")
        return None, image_paths_batch

# --- Main Script ---
def process_images_for_inference(image_folder):
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' not found.")
        return

    image_files = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"Found {len(image_files)} images in '{image_folder}'.")

    inference_results = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        current_batch_images_base64 = []
        current_batch_image_paths = []

        for idx, image_path in enumerate(tqdm(image_files, desc="Encoding and batching images")):
            encoded_image = encode_image_to_base64(image_path)
            if encoded_image:
                current_batch_images_base64.append(encoded_image)
                current_batch_image_paths.append(image_path)

            if len(current_batch_images_base64) == BATCH_SIZE or (idx == len(image_files) - 1):
                futures.append(executor.submit(
                    send_inference_request,
                    current_batch_images_base64.copy(),
                    current_batch_image_paths.copy()
                ))
                current_batch_images_base64.clear()
                current_batch_image_paths.clear()

        for future in as_completed(futures):
            result, original_paths = future.result()
            if result and 'predictions' in result:
                predictions = result['predictions']
                for i, pred in enumerate(predictions):
                    inference_results[original_paths[i]] = pred
            else:
                print(f"No valid predictions for batch starting with {original_paths[0] if original_paths else 'N/A'}")

    total_time = time.time() - start_time
    print(f"\nCompleted inference for {len(inference_results)} images in {total_time:.2f}s.")
    
    return inference_results

if __name__ == "__main__":
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # Create dummy images if folder is empty (for testing)
    if not os.listdir(IMAGE_FOLDER):
        from PIL import ImageDraw
        for i in range(500):
            img = Image.new('RGB', (100, 100), (i*40, i*30, i*20))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"Test {i}", fill=(255, 255, 0))
            img.save(os.path.join(IMAGE_FOLDER, f"test_image_{i}.jpg"))
        print("Dummy images created.")

    # ---- PERFORMANCE MEASURE ----
    start_time = time.time()
    results = process_images_for_inference(IMAGE_FOLDER)
    end_time = time.time()

    total_images = len(results) if results else 0
    total_time = end_time - start_time
    avg_latency = (total_time / total_images) if total_images else 0

    print(f"\nCompleted inference for {total_images} images in {total_time:.2f} seconds.")
    print(f"Average latency per image: {avg_latency:.4f} seconds.")
    results_file = "inference_results.jsonl"

    # ---- SAVE TO FILE ----
    if results:
        with open(results_file, "w") as f:
            for img_path, prediction in results.items():
                f.write(json.dumps({"image": os.path.basename(img_path), "prediction": prediction}) + "\n")
        print(f"Inference results saved to '{results_file}'.")
    else:
        print("No inference results obtained.")
