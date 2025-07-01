# app/generator.py
import requests
from .rag import retrieve_relevant_docs

def generate_response_deepseek(question, embedding_model, index, docs, api_key, max_tokens=300):
    docs_found = retrieve_relevant_docs(question, embedding_model, index, docs)
    context = "\n".join(docs_found)
    prompt = (
        f"Voici des informations sur Roger :\n{context}\n\n"
        f"Question : {question}\n"
        f"Réponse avec une taille raisonnable :"
    )


    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={

            #"model": "meta-llama/llama-3.3-8b-instruct:free",
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    )

    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content.strip(), context
