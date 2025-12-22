import os
import requests
import tarfile
import shutil
from io import BytesIO

# --- Configuration ---

BASE_V5 = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/"
MODEL_MAP = {
    "English": (
        "en_PP-OCRv5_mobile_rec",
        f"{BASE_V5}/en_PP-OCRv5_mobile_rec_infer.tar",
    ),
}

DEST_DIR = "/app/local_models"


def download_and_extract(name, model_name, url):
    print(f"[{name}] Downloading {model_name}...")

    final_path = os.path.join(DEST_DIR, model_name)

    # Check if we already have it
    if os.path.exists(final_path):
        print(f"Already exists: {final_path}")
        return

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Error: URL returned {response.status_code}")
            return

        with tarfile.open(fileobj=BytesIO(response.content), mode="r:tar") as tar:
            tar.extractall(path=DEST_DIR)

            # Rename it to "{model_name}" to keep paths clean
            extracted_name = f"{model_name}_infer"
            extracted_path = os.path.join(DEST_DIR, extracted_name)

            if os.path.exists(extracted_path) and extracted_path != final_path:
                shutil.move(extracted_path, final_path)
                print(f"Saved to {final_path}")
            elif os.path.exists(final_path):
                print(f"Saved to {final_path}")
            else:
                print(
                    f"Warning: Folder unpacked but name check failed. Check {DEST_DIR}"
                )

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    for lang, (model, url) in MODEL_MAP.items():
        download_and_extract(lang, model, url)
