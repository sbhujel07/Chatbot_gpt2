import requests
import json
import os


#Downloading the data from the github and save to the json format.
# Create folder if not exists
os.makedirs("data/raw", exist_ok=True)

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Save locally
with open("data/raw/instruction-data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Dataset saved to data/raw/instruction-data.json")



# 4️⃣ Split data
os.makedirs("data/processed", exist_ok=True)
train_portion = int(len(data)*0.85)
test_portion = int(len(data)*0.10)
validation_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
validation_data = data[train_portion + test_portion:]

# 5️⃣ Save splits as JSON
with open("data/processed/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("data/processed/val.json", "w") as f:
    json.dump(validation_data, f, indent=2)

with open("data/processed/test.json", "w") as f:
    json.dump(test_data, f, indent=2)

print("Train/Validation/Test splits saved in data/processed/")