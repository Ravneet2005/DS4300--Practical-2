import time
import json
import numpy as np
import redis
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import tiktoken
import psutil
import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

TOP_K = 3

def extract_text_from_pdf(pdf_path):
    " read PDF with pypdf."
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text


def chunk_text(text, chunk_size, overlap):
    " text chunking using tiktoken."
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks


def embed_chunks(model, chunks):
    " generate embeddings for text chunks."
    return model.encode(chunks, normalize_embeddings=True)


def embeddings_redis(redis_client, chunks, embeddings, chunk_size, overlap):
    " store chunk embeddings in Redis with metadata."
    prefix = f"chunk_{chunk_size}_{overlap}"

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"{prefix}:{i}"
        redis_client.hset(key, mapping={
            "text": chunk,
            "embedding": json.dumps(embedding.tolist()),
            "chunk_size": chunk_size,
            "overlap": overlap
        })


def search_redis(redis_client, query_embedding, chunk_size, overlap):
    " get most relevant chunks for a query using cosine similarity."
    prefix = f"chunk_{chunk_size}_{overlap}"

    results = []
    for key in redis_client.keys(f"{prefix}:*"):
        chunk_data = redis_client.hgetall(key)
        stored_embedding = np.array(json.loads(chunk_data["embedding"]))
        similarity = 1 - cosine(query_embedding, stored_embedding)
        results.append((chunk_data["text"], similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:TOP_K]


def embeddings_chroma(chroma_client, chunks, embeddings, chunk_size, overlap):
    "store chunk embeddings in Chroma."
    collection = chroma_client.get_or_create_collection(name=f"chunk_{chunk_size}_{overlap}")
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings.tolist())


def search_chroma(chroma_client, query_embedding, chunk_size, overlap):
    " retrieve most relevant chunks for a query using Chroma."
    collection = chroma_client.get_collection(name=f"chunk_{chunk_size}_{overlap}")
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=TOP_K)
    return list(zip(results["documents"][0], results["distances"][0]))


def embeddings_qdrant(qdrant_client, chunks, embeddings, chunk_size, overlap):
    "store chunk embeddings in Qdrant."
    collection_name = f"chunk_{chunk_size}_{overlap}"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"text": chunk, "chunk_size": chunk_size, "overlap": overlap}
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)


def search_qdrant(qdrant_client, query_embedding, chunk_size, overlap):
    "retrieve most relevant chunks using Qdrant."
    collection_name = f"chunk_{chunk_size}_{overlap}"
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=TOP_K
    )
    return [(result.payload["text"], result.score) for result in results]


def get_memory_usage():
    "get current memory usage"
    process = psutil.Process(os.getpid())
    # convert to MB
    return process.memory_info().rss / (1024 * 1024)


def write_to_file(file, message):
    "write a message to the output file."
    with open(file, "a") as f:
        f.write(message + "\n")

