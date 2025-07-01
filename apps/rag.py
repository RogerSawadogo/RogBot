from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def create_index(data_path='data/data.txt'):
    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Chargement des documents...")
    with open(data_path, 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f if line.strip()]

    print(f"Encodage de {len(docs)} documents...")
    embeddings = embedding_model.encode(docs)

    print("Création de l'index FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    print("Sauvegarde de l'index et des documents...")
    faiss.write_index(index, 'index.faiss')
    with open('docs.pkl', 'wb') as f:
        pickle.dump(docs, f)

    print("Indexation terminée.")

def load_index_and_docs():
    print("Chargement de l'index FAISS et des documents...")
    index = faiss.read_index('index.faiss')
    with open('docs.pkl', 'rb') as f:
        docs = pickle.load(f)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model, index, docs

def retrieve_relevant_docs(question, embedding_model, index, docs, top_k=3):
    question_vec = embedding_model.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    return [docs[i] for i in I[0]]
