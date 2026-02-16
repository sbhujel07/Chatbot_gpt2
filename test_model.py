#test if the model load correctly
import torch
from configs.config import NEW_CONFIG
from Src.model import GPTModel

model = GPTModel(NEW_CONFIG)
model.eval()

vocab_size = NEW_CONFIG["vocab_size"]
x = torch.randint(0, vocab_size, (2, 5))
logits = model(x)

print("Input Shape:",x.shape)
print("Output Shape:",logits.shape)
