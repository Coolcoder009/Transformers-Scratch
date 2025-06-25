# 🔤 English–Tamil Transformer from Scratch (PyTorch)

This project demonstrates a **Transformer-based neural machine translation** model built **completely from scratch** using **PyTorch**. The goal was to translate English to Tamil using a small dataset and limited compute resources.

---

## 📌 Overview

- ✅ Implemented Transformer architecture from scratch (no high-level libs like `torch.nn.Transformer`)
- 🗣️ Trained on **200,000 English–Tamil sentence pairs**
- 🏋️‍♂️ Trained for **20 epochs** due to compute limitations
- 🔄 Translation supported via:
  - **Greedy decoding**
  - **Beam search decoding**
- 📉 Accuracy is low due to limited training, but the model shows learning ability
- 🚀 When trained for more epochs, performance improves significantly

---

## 🧠 Model Architecture

- Encoder–Decoder Transformer
- Positional Encoding
- Masked Multi-Head Attention
- Layer Normalization
- Cross-Attention between Encoder and Decoder
- Beam Search and Greedy Decoding

---

## ⚙️ Training Details

| Item              | Description                    |
|-------------------|--------------------------------|
| Dataset           | 200,000 English–Tamil pairs    |
| Epochs            | 20                             |
| Batch Size        | 32                             |
| Optimizer         | Adam with warmup scheduler     |
| Loss              | Cross-Entropy with masking     |
| Beam Width        | 3                              |

---

## 📥 Example Translations

| English                     | Tamil (Greedy)                              | Tamil (Beam Search)                 |
|-----------------------------|---------------------------------------------|-------------------------------------|
| It's not your fault         | உன் முழுவதும் இல்லை                         | உன் முழுவதும் இல்லை                 |
| How are you brother?        | உங்கள் காதலிக்கு என்ன செய்து கொண்டார்கள்?     | உங்களுக்கு நீங்கள் என்ன?             |

> ⚠️ These translations are **not accurate yet** due to limited training. With more epochs, the model's performance improves.

---

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/english-tamil-transformer.git
cd english-tamil-transformer

# Train the model
python train.py
