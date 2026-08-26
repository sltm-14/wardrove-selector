import os
os.environ["KAGGLEHUB_CACHE"] = "./data"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

print("Path to dataset files:", path)