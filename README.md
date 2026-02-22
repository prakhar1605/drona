<div align="center">

# 🎯 DRONA AI
### Autonomous Interview Agent

**adaptive · memory-driven · real-time**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-dronaai.in-7c6aff?style=for-the-badge&logo=streamlit)](https://dronaai.in)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Memory-4fffb0?style=for-the-badge)](https://trychroma.com)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)

<br/>

> AI-powered interview simulation platform that **remembers your weaknesses**, **adapts difficulty in real-time**, and delivers **personalized feedback** — built for students preparing for technical interviews.

</div>

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🧠 **Interview Memory** | ChromaDB vector embeddings store past performance and personalize future sessions |
| ⚡ **Adaptive Difficulty** | Auto-adjusts question difficulty using a rolling 3-answer performance window |
| 🌊 **Streaming Responses** | Token-by-token real-time AI feedback — no waiting |
| 💾 **Redis Caching** | Frequent question sets cached to reduce LLM latency and API cost |
| 📄 **Resume-Aware** | Upload PDF resume → RAG pipeline generates personalized questions |
| 🎯 **Role-Targeted** | Questions tailored to SWE, Data Scientist, DevOps, AI Engineer, and more |
| 📊 **Performance Analytics** | Topic-wise breakdown, weak area detection, downloadable report |

---

## 🏗️ Architecture

```
User (Browser)
      ↓
  Streamlit Frontend (app.py)
      ↓
  ┌─────────────────────────────────────┐
  │            Backend Layer            │
  │                                     │
  │  llm.py       → OpenRouter API     │
  │                 (GPT-4o-mini)       │
  │                                     │
  │  memory.py    → ChromaDB           │
  │                 (vector embeddings) │
  │                                     │
  │  cache.py     → Redis Cloud        │
  │                 (question caching)  │
  │                                     │
  │  adaptive.py  → Difficulty Engine  │
  │                 (score-based logic) │
  │                                     │
  │  pdf_parser.py → Resume Parser     │
  │                  (RAG context)      │
  └─────────────────────────────────────┘
```

---

## 🚀 Tech Stack

- **Frontend** — Streamlit
- **LLM** — GPT-4o-mini via OpenRouter (streaming)
- **Vector DB** — ChromaDB (local persistent embeddings)
- **Cache** — Redis Cloud (TTL-based question caching)
- **PDF Parsing** — pypdf (resume text extraction)
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/dronaai.git
cd dronaai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
REDIS_URL=redis://default:password@host:port
```

- Get OpenRouter key → [openrouter.ai](https://openrouter.ai)
- Get Redis free tier → [redis.io/try-free](https://redis.io/try-free)

### 4. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

---

## 📁 Project Structure

```
dronaai/
├── app.py                  ← Main Streamlit app
├── backend/
│   ├── llm.py              ← LLM calls + streaming
│   ├── memory.py           ← ChromaDB vector memory
│   ├── cache.py            ← Redis caching layer
│   ├── adaptive.py         ← Difficulty engine
│   └── pdf_parser.py       ← PDF resume parser
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 How the Memory System Works

```
Session 1:
  User answers Graphs question → Score 20% → stored as embedding in ChromaDB

Session 2:
  ChromaDB queried for weak areas → "Graphs" retrieved
  Next interview → 40% more Graphs questions generated
  Difficulty auto-adjusted based on rolling score window
```

---

## 📈 Adaptive Difficulty Logic

```python
recent_3_answers = answers[-3:]
avg_score = mean(scores)

if avg_score > 8:   → Tough     # crushing it 🔥
elif avg_score > 5: → Moderate  # doing well 👍
else:               → Easy      # needs support 📚
```

---

## 🌐 Live Demo

👉 **[dronaai.in](https://dronaai.in)**

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">

Built with ❤️ by **Prakhar Pandey**

</div>
