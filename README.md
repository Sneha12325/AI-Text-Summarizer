# 🧠 AI Text Summarizer

An AI-powered web application that summarizes long text passages using a **Transformer model (BART-Large-CNN)**.  
Originally built with **Flask**, this project was later optimized with **FastAPI + Gradio** for faster deployment and smoother inference.

---

## 🚀 Demo
👉 **[Live App on Hugging Face](https://huggingface.co/spaces/Sneha7676P/ai-summarizer-fast)**  
*(FastAPI + Gradio version — loads under 10 seconds!)*

---

## 🧩 Project Versions

### 1️⃣ Flask Version (Full UI/UX)
- Complete backend architecture (600+ lines of production-ready Flask code)
- Custom HTML/CSS/JS templates for better user experience
- Includes:
  - Redis caching
  - Logging and rate limiting
  - Modularized routes and configuration
- Great for demonstrating **full-stack backend design skills**

### 2️⃣ FastAPI + Gradio Version (Optimized for Speed)
- Lightweight rewrite focused on **performance and simplicity**
- Built with **FastAPI** backend and **Gradio** interface
- Deploys instantly on Hugging Face Spaces
- Simplified dependencies → no Redis, no template rendering

---

## ⚙️ Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend/UI** | Flask Templates (HTML, CSS, JS), Gradio |
| **Backend** | Flask → FastAPI |
| **Model** | Hugging Face Transformers (BART-Large-CNN) |
| **Deployment** | Hugging Face Spaces |
| **Optional Tools** | Redis, Gunicorn, Pipenv, Logging, Caching |

---

## 🧠 Model Used

- **Model:** `facebook/bart-large-cnn`  
- **Task:** Text Summarization  
- The model takes long text and outputs a concise summary while preserving key context.

---

## 💡 Key Features

- Summarize large paragraphs or documents in seconds  
- Clean and minimal UI (Gradio) for fast testing  
- Full production-grade Flask version available for code review  
- Hugging Face integration for easy public access  

## ⚙️ How to Run Locally

```bash
# 1. Clone this repository
git clone https://huggingface.co/spaces/Sneha7676P/ai-summarizer-fast
cd ai-summarizer-fast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
Then visit: http://127.0.0.1:7860
