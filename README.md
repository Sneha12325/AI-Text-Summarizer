# 🧠 AI Text Summarizer (Fast Version)

A lightning-fast **AI-powered text summarization app** built using **Gradio + FastAPI + Transformers**.  
This version is optimized for deployment speed and simplicity — perfect for demos, recruiters, and backend showcases.  

🚀 **Live Demo:** [AI Summarizer on Hugging Face](https://huggingface.co/spaces/Sneha7676P/ai-summarizer-fast)

---

## 💡 About the Project

This app takes long pieces of text and generates concise, meaningful summaries using a transformer-based model from Hugging Face (`facebook/bart-large-cnn`).

### 🔍 Features
- ⚡ **Fast API response** using Gradio + FastAPI combo  
- 🧠 **Transformer-based summarization** (BART model)  
- 🖥️ **Clean minimal UI** powered by Gradio  
- ☁️ **Fully deployed on Hugging Face Spaces**  
- 🧩 **Easily extendable** for multi-language or abstractive summarization  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | FastAPI |
| UI | Gradio |
| ML Model | Facebook BART-Large-CNN |
| Hosting | Hugging Face Spaces |
| Version Control | Git + GitHub + HF Hub |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone this repository
git clone https://huggingface.co/spaces/Sneha7676P/ai-summarizer-fast
cd ai-summarizer-fast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
