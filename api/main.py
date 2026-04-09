#load the finetuned model 
import os
import torch
from src.model import GPTModel
from configs.config import NEW_CONFIG
from src.device import device

# Path to model (mount this folder)
model_path = os.getenv("MODEL_PATH", "./models")  # default if not provided
model_file = os.path.join(model_path, "gpt2-medium 355M-sft.pth")


#Initialize the model
model = GPTModel(NEW_CONFIG)


model.load_state_dict(torch.load("models/gpt2-medium 355M-sft.pth",map_location=device))

#Move to cpu/gpu
model.to(device)
model.eval()
# print("Finetuned model Loaded Succesfully")