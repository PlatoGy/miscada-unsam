import requests

# API URL (modify according to your service's running address)
API_URL = "http://localhost:8000/point_unsam"

# Path to the image to be uploaded (must be an RGB image)
IMAGE_PATH = "docs/demos/liver.png"
# Sending the request
with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

# Handling the response
if response.status_code == 200:
    result = response.json()
    print("Inference successful! Image link returned:")
    print("Marked input image:", result["marked_image"])
    print("Segmentation results:")
    for url in result["segmentation_results"]:
        print(" -", url)
else:
    print("Request failed:", response.status_code)
    print(response.text)
