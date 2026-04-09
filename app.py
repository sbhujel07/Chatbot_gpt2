from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

    # Simple HTML UI
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>GPT-2 Chat UI</title>
        </head>
        <body>
            <h2>Chat with GPT-2</h2>
            <form id="chatForm">
                Instruction:<br>
                <input type="text" id="instruction" name="instruction" size="50"><br><br>
                Input Text:<br>
                <input type="text" id="input_text" name="input_text" size="50"><br><br>
                <input type="submit" value="Send">
            </form>
            <h3>Response:</h3>
            <pre id="response"></pre>

            <script>
                const form = document.getElementById('chatForm');
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const instruction = document.getElementById('instruction').value;
                    const input_text = document.getElementById('input_text').value;

                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({instruction, input_text})
                    });
                    const data = await res.json();
                    document.getElementById('response').textContent = data.response;
                });
            </script>
        </body>
    </html>
    """