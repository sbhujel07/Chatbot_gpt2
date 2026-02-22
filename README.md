# 🤖 GPT-2 Chatbot – Built From Scratch

## 📌 About This Project

This project started as a challenge:  
"Can I implement GPT-2 myself instead of relying completely on libraries?"

So instead of using HuggingFace’s ready-made model, I implemented the GPT-2 architecture from scratch — carefully matching the original structure — then loaded pretrained GPT-2 weights and fine-tuned the model on an Alpaca-style instruction dataset to make it conversational.

This repository represents my deep dive into Transformer architecture, weight mapping, and instruction tuning.

---

## 🧠 Model Architecture

The model follows the original GPT-2 (decoder-only Transformer) design:

- Token + positional embeddings
- Multi-head self-attention
- Causal masking
- Shortcut connections
- Layer Normalization
- Feed-forward network (MLP block)

Configuration used:

- Vocabulary size: 50257  
- Context length: 1024  
- Embedding dimension: 1024  
- Attention heads: 16  
- Transformer layers: 24
- Drop_rate: 0.1
- qkv_bias: True  

The goal was to match GPT-2 Medium architecture exactly so pretrained weights could be loaded without mismatch.


## 🚀 Training Journey

### 🔹 Step 1 – Architecture Implementation

I first implemented:
- Attention mechanism
- Transformer block
- Full GPT model class
- Proper weight initialization



### 🔹 Step 2 – Loading Pretrained Weights

Instead of training from zero, I:
- Loaded official GPT-2 pretrained weights
- Carefully mapped parameter names
- Verified tensor shapes
- Tested forward pass to confirm correctness

This ensured my implementation was 100% compatible.


### 🔹 Step 3 – Instruction Fine-Tuning

After confirming the pretrained model worked correctly:

- Dataset: Alpaca-style instruction dataset  
- Objective: Supervised Fine-Tuning (SFT)  
- Loss: Cross-Entropy  
- Optimizer: AdamW  

The model was trained to generate helpful, instruction-following responses.
