import os
import urllib.request

# Updated URLs for the model files
model_url = "https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel"
prototxt_url = "https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt"

# File names
model_file = "MobileNetSSD_deploy.caffemodel"
prototxt_file = "MobileNetSSD_deploy.prototxt"

# Function to download a file if it doesn't exist
def download_file(url, file_name):
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded {file_name}.")
    else:
        print(f"{file_name} already exists.")

# Download the model and prototxt files
download_file(model_url, model_file)
download_file(prototxt_url, prototxt_file)
