#load the finetuned model 
import torch
from src.model import GPTModel
from configs.config import NEW_CONFIG
from src.device import device

#Initialize the model
model = GPTModel(NEW_CONFIG)


model.load_state_dict(torch.load("models/gpt2-medium 355M-sft.pth",map_location=device))

#Move to cpu/gpu
model.to(device)
model.eval()
# print("Finetuned model Loaded Succesfully")