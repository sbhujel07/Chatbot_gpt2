🚀 GPT-2 Chatbot – Built From Scratch
📌 About This Project

This project began with a simple but powerful question:

"Can I build GPT-2 from scratch instead of relying entirely on libraries?"

Instead of using prebuilt implementations like HuggingFace, I implemented the GPT-2 architecture from the ground up, carefully reproducing its internal structure. After building the model, I loaded pretrained GPT-2 weights and fine-tuned it on an Alpaca-style instruction dataset to create a conversational chatbot.

This project reflects a deep hands-on understanding of:

Transformer architecture
Weight mapping & compatibility
Instruction fine-tuning (SFT)
🧠 Model Architecture

The model strictly follows the GPT-2 (decoder-only Transformer) design:

Core Components:
Token Embeddings + Positional Embeddings
Multi-Head Self-Attention
Causal Masking (for autoregressive generation)
Residual (Shortcut) Connections
Layer Normalization
Feed-Forward Network (MLP Block)
⚙️ Configuration (GPT-2 Medium Equivalent)
Parameter	Value
Vocabulary Size	50,257
Context Length	1024
Embedding Dimension	1024
Attention Heads	16
Transformer Layers	24
Dropout Rate	0.1
QKV Bias	True

✅ The architecture was matched exactly to GPT-2 Medium to ensure seamless pretrained weight loading.

🛠️ Development Journey
🔹 Step 1 – Building the Architecture

Implemented from scratch:

Self-attention mechanism
Transformer block
Full GPT model class
Weight initialization
🔹 Step 2 – Loading Pretrained Weights

Instead of training from scratch (which is computationally expensive):

Loaded official GPT-2 pretrained weights
Mapped parameter names manually
Verified tensor shapes
Validated forward pass outputs

✅ This ensured full compatibility with pretrained GPT-2.

🔹 Step 3 – Instruction Fine-Tuning

After validating the base model:

Dataset: Alpaca-style instruction dataset
Training Type: Supervised Fine-Tuning (SFT)
Loss Function: Cross-Entropy
Optimizer: AdamW

🎯 Goal: Make the model generate helpful, instruction-following responses.

⚡ FastAPI Chatbot API

The project includes a FastAPI backend to serve the fine-tuned chatbot.

🔧 Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install dependencies
pip install -r requirements.txt
3. Run the server
uvicorn app:app --reload
4. Open in browser
http://127.0.0.1:8000/docs