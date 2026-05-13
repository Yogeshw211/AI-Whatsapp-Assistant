import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_data():
    with open("data/products.json", "r", encoding="utf-8") as file:
        products = json.load(file)
    return products

def get_embedding(text):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(1536, dtype="float32")

def build_index(products):
    vectors = []
    for row in products:
        text = f"Product: {row['name']} Category: {row['category']} Age: {row.get('age', '')} Desc: {row.get('description', '')}"
        vec = get_embedding(text)
        vectors.append(vec)

    vectors = np.vstack(vectors)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def is_age_compatible(requested_age, product_age_range):
    """Handles numeric ranges and the 'Mum' category safely."""
    if product_age_range == "Mum":
        return False 
    
    try:
        if "-" in str(product_age_range):
            min_age, max_age = map(int, str(product_age_range).split("-"))
            return min_age <= requested_age <= max_age
        return int(product_age_range) == requested_age
    except:
        return False

def extract_age(query):
    query = query.lower()
    match = re.search(r'(\d+)', query)
    if match:
        val = int(match.group(1))
        if "month" in query: return val / 12
        return val
    return None


def detect_category(query):
    query = query.lower()

    category_map = {
        "Toys": ["toy", "play", "jumper", "walker", "teether", "musical"],
        "Gear": ["stroller", "car seat", "travel"],
        "Feeding": ["milk", "bottle", "pump", "food maker", "brezza"],
        "Diapers": ["wipes", "diaper", "pampers"],
        "Bath": ["shampoo", "gel", "lotion", "bathtub"],
        "Nursery": ["crib", "basket", "nest", "pail"]
    }

    for category, keywords in category_map.items():
        if any(word in query for word in keywords):
            return category

    return None


def search_products(query, products, index, vectors, limit=5):
    detected_cat = detect_category(query)

    if detected_cat:
        filtered_products = [p for p in products if p.get('category', '').lower() == detected_cat.lower()]
    else:
        filtered_products = products
        
    if not filtered_products:
        return []

    query_vec = get_embedding(query).reshape(1, -1)

    results = []
    for p in filtered_products:
        original_idx = products.index(p)
        p_vector = vectors[original_idx]
        score = np.dot(query_vec, p_vector)
        results.append((score, p))
    results.sort(key=lambda x: x[0], reverse=True)
    
    return [r[1] for r in results[:limit]]

def generate_response(query, products):
    if not products:
        return "I couldn't find any items matching that. Try a different age or budget!"

    response_lines = ["Here are the top picks for you:"]
    for i, p in enumerate(products, start=1):
        line = f"{i}. **{p['name']}** - {p['price']} AED\n   _{p['description']}_"
        response_lines.append(line)

    return "\n\n".join(response_lines)