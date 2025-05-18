# Filename_SQUATCH

# Keyphrase Generation and Abstractive Summarization using Pegasus-XSum with LoRA

This project focuses on performing **abstractive summarization** and **keyphrase generation** on scientific abstracts using the `google/pegasus-xsum` model. We fine-tune the model efficiently using **Low-Rank Adaptation (LoRA)**, making it suitable for training on consumer-grade hardware such as Google Colab.

---

## 📚 Dataset

We use the **MIDAS/INSPEC (raw)** dataset, which contains tokenized scientific abstracts annotated with human-generated keyphrases.

- Each abstract is reconstructed into natural language format before feeding into the model.
- For keyphrase generation, we optionally prefix inputs with `"keyphrases:"` to guide the model output.

---

## 🧠 Model

We utilize the **Pegasus-XSum** model from Hugging Face, which is pretrained for extreme summarization.

### Configuration:
- **Input length**: Up to 512 tokens
- **Output length**:
  - Summaries: Max 64 tokens
  - Keyphrases: Max 32 tokens
- **Batch generation**: 8 for inference

---

## 🛠 Fine-tuning Setup

We use **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** to minimize memory and compute requirements.

### LoRA Settings:
- `r = 8` (rank)
- `lora_alpha = 32` (scaling factor)
- `target_modules = ["q_proj", "v_proj"]`
- `lora_dropout = 0.1`
- `task_type = SEQ_2_SEQ_LM`

Only the attention projection layers are trained, with fewer than 1M parameters updated.

---

## ⚙️ Training Details

- **Trainer**: `Seq2SeqTrainer` from Hugging Face
- **Epochs**: 16
- **Learning rate**: `1e-5`
- **Optimizer**: AdamW with weight decay `0.01`
- **Per-device batch size**: 4
- **Gradient accumulation steps**: 8 (simulates batch size of 32)
- **Mixed precision**: `fp16=True` for reduced memory use and faster training
---

## 📊 Evaluation Metrics

We use **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** scores from Hugging Face's `evaluate` library to assess model performance based on overlap with reference outputs.

---

## 💻 Libraries and Tools

- `transformers`
- `datasets`
- `evaluate`
- `peft`
- `accelerate`
- `torch` (PyTorch)
- Google Colab (GPU-enabled environment)

---
