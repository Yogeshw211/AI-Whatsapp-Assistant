# AI Product Recommendation & Semantic Search Assistant

An AI-powered semantic product search and recommendation assistant built using FastAPI, OpenAI Embeddings, and FAISS.

The system understands natural language queries and returns intelligent, context-aware product recommendations using semantic similarity search and filtering techniques.


# Features

- Semantic product search using AI embeddings
- Context-aware recommendations
- Strict category filtering
- Price-intent understanding
- Age-safety validation
- Fast similarity search with FAISS
- Follow-up query handling
- REST API using FastAPI
- Lightweight frontend using Vanilla JavaScript


# Tech Stack

- Python
- FastAPI
- OpenAI Embeddings
- FAISS
- Pandas
- NumPy
- Vanilla JavaScript
- HTML/CSS


# Project Structure

project/

├── app.py

├── requirements.txt

├── .env

├── README.md

├── static/

├── templates/

├── data/

└── embeddings/



# Installation Guide

## 1. Clone the Repository

git clone https://github.com/yourusername/your-repo-name.git

cd your-repo-name

---

## 2. Create Virtual Environment

python -m venv venv

### Activate Environment

### Windows

venv\Scripts\activate

### Mac/Linux

source venv/bin/activate


## 3. Install Dependencies

pip install -r requirements.txt


## 4. Add Environment Variables

Create a `.env` file and add:

OPENAI_API_KEY=your_api_key_here


## 5. Run the Application

uvicorn app:app --reload

Server starts at:

http://127.0.0.1:8000


# API Endpoint

## POST /chat

Example Request:

{
  "query": "best affordable stroller"
}


# How It Works

1. User enters a natural language query
2. OpenAI embeddings convert the query into vectors
3. FAISS performs semantic similarity matching
4. Filters are applied:
   - Category filtering
   - Price intent
   - Safety validation
5. Best matching products are returned


# Future Improvements

- Multi-language support
- Voice-based search
- Personalized recommendations
- User authentication
- Analytics dashboard
- Real-time product scraping


# Contributing

Pull requests are welcome.

For major changes, please open an issue first to discuss improvements.


# License

This project is licensed under the MIT License.


# Author

Developed by Your Name