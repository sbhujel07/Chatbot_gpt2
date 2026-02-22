import requests

# API endpoint
url = "http://localhost:8000/chat"

# Example request
data = {
    "instruction": "Identify the correct spelling: 'recieve' or 'receive'.",
    "input_text": ""
}

response = requests.post(url, json=data)

# Print the model response
print("Model Response:\n", response.json()["response"])