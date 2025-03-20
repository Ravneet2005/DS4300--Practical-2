import time
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http import models

PDF_PATH = "module3-6.pdf"
CHUNK_SIZES = [500]
CHUNK_OVERLAPS = [50]
EMBEDDING_MODEL = "hkunlp/instructor-xl"
TOP_K = 3
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "notes"

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768,  # Embedding size for hkunlp/instructor-xl
            distance=models.Distance.COSINE,
        ),
    )

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text


def chunk_text(text, chunk_size, overlap):

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks


def embed_and_store_chunks(chunks, chunk_size, overlap):

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)


    points = [
        models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def retrieve_relevant_chunks(query, top_k=TOP_K):

    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
    )

    return [hit.payload["text"] for hit in results] if results else []


def generate_response(model, context, query):

    prompt = f"Given the context:\n{context}\n\nAnswer the following question:\n{query}"

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


if __name__ == "__main__":

    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text)} characters from PDF\n")

    for chunk_size in CHUNK_SIZES:
        for overlap in CHUNK_OVERLAPS:
            chunks = chunk_text(text, chunk_size, overlap)

            start_time = time.time()
            embed_and_store_chunks(chunks, chunk_size, overlap)

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        retrieved_chunks = retrieve_relevant_chunks(query)

        if retrieved_chunks:

            response = generate_response("mistral", "\n".join(retrieved_chunks), query)
            print("\nMistral Response:\n", response)
        else:
            print("No relevant chunks found for the query.")