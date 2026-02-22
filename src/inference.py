import torch
from api.main import model
from src.data_formating import format_input
from data.processed.test_data import test_data
from src.tokenizer import tokenizer
from src.tokenizer import text_to_token_ids
from src.tokenizer import token_ids_to_text
from src.device import device
from configs.config import NEW_CONFIG
from src.generate import Generate_text



#Turn off drop out and batch normalization
model.eval()

#see 3 responses from the test data and model response on the same input.
torch.manual_seed(123)
for data in test_data[:3]:

  input_text = format_input(data)
  idx = text_to_token_ids(input_text,tokenizer).to(device)
  with torch.no_grad():
    token_id = Generate_text(
        model = model,
        idx = idx,
        max_new_tokens=50,
        context_size=NEW_CONFIG["context_length"],
        top_k=1,
        temperature=0.0,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_id,tokenizer)
    model_response = generated_text[len(input_text):].replace("###Response:","").strip()

    print(input_text)
    print("\nModel Response:\n>>",model_response)
    print("\nAccurate Response:\n>>",data["output"])
    print("------------------------------------")
