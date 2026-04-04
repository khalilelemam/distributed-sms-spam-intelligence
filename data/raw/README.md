# Raw Data Folder

This project runs in Colab and downloads data directly from Kaggle using kagglehub.

Dataset link:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Download snippet used in the notebooks:

import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)

Notes:
- You do not need to manually place raw files in this folder when running in Colab.
- If you run locally, keep raw data out of Git history.
