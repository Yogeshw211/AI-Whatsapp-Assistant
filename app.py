import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import uuid
from fastapi.staticfiles import StaticFiles


# Importing RAG pipeline
from rag.pipeline import load_data, build_index, search_products, generate_response


load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sessions = {}

def get_session_data(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {"chat_history": [], "last_products": []}
    return sessions[session_id]



df = load_data()
index, vectors = build_index(df)

class QueryRequest(BaseModel):
    session_id: str = None  
    age: str
    budget: str
    purpose: str
    country: str



def detect_price_intent(query):
    query = query.lower()

    if any(word in query for word in ["cheap", "cheapest", "low price", "budget"]):
        return "low"

    if any(word in query for word in ["expensive", "premium", "high end", "best"]):
        return "high"

    return None


@app.get("/")
def home():
    return {"message": "Mumzworld AI Assistant is running 🚀"}


@app.post("/chat")
async def chat(request: QueryRequest):

    session_id = request.session_id or str(uuid.uuid4())
    session_data = get_session_data(session_id)
    
    query = f"I am looking for {request.purpose} for a {request.age} year old with a budget of {request.budget} in {request.country}"

    try:
       
        limit = 10 if "more" in query else 5
        products = search_products(query, df, index, vectors, limit=limit)

        if not products:
            return {
                "session_id": session_id,
                "response": "I couldn't find any products matching those specific criteria. Could you try adjusting the age or budget?",
                "products": []
            }

        answer = generate_response(query, products)

        
        session_data["last_products"] = products
        session_data["chat_history"].append({"user": query, "bot": answer})

        return {
            "session_id": session_id,
            "response": answer,
            "products": products
        }

    except Exception as e:
        
        print(f"Error encountered: {e}")
        return {
            "session_id": session_id,
            "response": "I'm having a little trouble reaching my AI brain right now. Please try again in a moment! 🍼",
            "products": []
        }
