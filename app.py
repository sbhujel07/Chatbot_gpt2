from fastapi import FastAPI
from pydantic import BaseModel
import torch
from api.main import model
from src.tokenizer import tokenizer, text_to_token_ids, token_ids_to_text
from src.generate import Generate_text
from configs.config import NEW_CONFIG
from src.device import device


torch.manual_seed(123)
model.eval()
# -------------------------------

app = FastAPI()

# Request model
class Request(BaseModel):
    instruction: str
    input_text: str = ""

@app.post("/chat")
def chat(request: Request):
    prompt = f"###Instruction:\n{request.instruction}\n###Input:\n{request.input_text}\n###Response:"

    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        token_id = Generate_text(
            model=model,
            idx=idx,
            max_new_tokens=50,
            context_size=NEW_CONFIG["context_length"],
            top_k=1,           # deterministic
            temperature=0.0,    # deterministic
            eos_id=50256
        )
    generated_text = token_ids_to_text(token_id, tokenizer)
    response = generated_text[len(prompt):].strip()
    return {"response": response}