from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder
from datetime import datetime, timezone
import hashlib, os, numpy as np
from torch import nn
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# -------------------- CONFIG --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "semantic-search-test"
NAMESPACE = "default"
ENGINE = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=False)
index = pc.Index(INDEX_NAME)
cross_encoder = CrossEncoder('models\ms-marco-MiniLM-L6-v2', num_labels=1)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -------------------- HELPERS --------------------

def my_hash(s):
    return hashlib.md5(s.encode()).hexdigest()

def get_embeddings(texts, engine=ENGINE):
    response = client.embeddings.create(input=texts, model=engine)
    return [d.embedding for d in response.data]

def get_embedding(text, engine=ENGINE):
    return get_embeddings([text], engine)[0]

def query_from_pinecone(query, top_k=5, include_metadata=True):
    query_embedding = get_embedding(query, engine=ENGINE)
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=include_metadata
    ).get('matches', [])

def get_results_from_pinecone(query, top_k=5, re_rank_model=None):
    results_from_pinecone = query_from_pinecone(query, top_k=top_k)
    if not results_from_pinecone:
        return []

    if re_rank_model is not None:
        sentence_pairs = [[query, r['metadata']['text']] for r in results_from_pinecone]
        scores = re_rank_model.predict(sentence_pairs, activation_fn=nn.Sigmoid())
        sorted_idx = np.argsort(scores)[::-1]
        reranked = [
            {
                "score": float(scores[i]),
                "text": results_from_pinecone[i]['metadata']['text'],
                "id": results_from_pinecone[i]['id'],
                "retrieval_score": results_from_pinecone[i]['score']
            }
            for i in sorted_idx
        ]
        return reranked
    return results_from_pinecone

# -------------------- ROUTES --------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = get_results_from_pinecone(query, re_rank_model=cross_encoder)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})
