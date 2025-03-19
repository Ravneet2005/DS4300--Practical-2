from ingest import *

def main():

    PDF_PATH = "module3-6.pdf"
    CHUNK_SIZES = [200, 500, 1000]
    CHUNK_OVERLAPS = [0, 50, 100]
    EMBEDDING_MODELS = ["thenlper/gte-base", "hkunlp/instructor-xl", "sentence-transformers/all-mpnet-base-v2"]
    VECTOR_DBS = ["redis", "chroma", "qdrant"]
    OUTPUT_FILE = "results.txt"

    # extract text from pdfs
    text = extract_text_from_pdf(PDF_PATH)
    write_to_file(OUTPUT_FILE, f"Extracted text length: {len(text)} characters\n")

    # intialize our vector databases
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    chroma_client = chromadb.Client()
    qdrant_client = QdrantClient(host="localhost", port=6333)

    metrics = {
        "chunk_sizes": CHUNK_SIZES,
        "chunk_overlaps": CHUNK_OVERLAPS,
        "embedding_models": EMBEDDING_MODELS,
        "vector_dbs": VECTOR_DBS,
        "results": []
    }

    # iterating over embedding models
    for model_name in EMBEDDING_MODELS:
        write_to_file(OUTPUT_FILE, f"\nUsing embedding model: {model_name}")
        embedding_model = SentenceTransformer(model_name)

        # iterating over vector databases
        for vector_db in VECTOR_DBS:
            write_to_file(OUTPUT_FILE, f"\nUsing vector database: {vector_db}")

            # diff chunk sizes and overlaps
            for chunk_size in CHUNK_SIZES:
                for overlap in CHUNK_OVERLAPS:
                    write_to_file(OUTPUT_FILE, f"\nProcessing with chunk_size={chunk_size}, overlap={overlap}")

                    # chunking
                    start_time = time.time()
                    chunks = chunk_text(text, chunk_size, overlap)
                    chunking_time = time.time() - start_time

                    # generate embeddings
                    start_time = time.time()
                    embeddings = embed_chunks(embedding_model, chunks)
                    embedding_time = time.time() - start_time

                    # memory usage
                    memory_usage = get_memory_usage()

                    # store and search in vector database
                    if vector_db == "redis":
                        redis_client.flushdb()
                        start_time = time.time()
                        embeddings_redis(redis_client, chunks, embeddings, chunk_size, overlap)
                        db_time = time.time() - start_time
                    elif vector_db == "chroma":
                        start_time = time.time()
                        embeddings_chroma(chroma_client, chunks, embeddings, chunk_size, overlap)
                        db_time = time.time() - start_time
                    elif vector_db == "qdrant":
                        start_time = time.time()
                        embeddings_qdrant(qdrant_client, chunks, embeddings, chunk_size, overlap)
                        db_time = time.time() - start_time

                    # metrics output
                    metrics["results"].append({
                        "model": model_name,
                        "vector_db": vector_db,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "num_chunks": len(chunks),
                        "chunking_time": chunking_time,
                        "embedding_time": embedding_time,
                        "db_time": db_time,
                        "memory_usage_mb": memory_usage
                    })

                    write_to_file(OUTPUT_FILE, f" - Stored {len(chunks)} chunks in {vector_db}")
                    write_to_file(OUTPUT_FILE, f" - Chunking time: {chunking_time:.2f} sec")
                    write_to_file(OUTPUT_FILE, f" - Embedding time: {embedding_time:.2f} sec")
                    write_to_file(OUTPUT_FILE, f" - {vector_db} storage time: {db_time:.2f} sec")
                    write_to_file(OUTPUT_FILE, f" - Memory usage: {memory_usage:.2f} MB")

    # metrics into a json
    with open("pipeline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # query tests
    query = "what is indexing?"
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    write_to_file(OUTPUT_FILE, "\nQUERY RESULTS:\n")
    for vector_db in VECTOR_DBS:
        for chunk_size in CHUNK_SIZES:
            for overlap in CHUNK_OVERLAPS:
                write_to_file(OUTPUT_FILE, f"\nVector DB: {vector_db}, Chunk Size: {chunk_size}, Overlap: {overlap}")

                if vector_db == "redis":
                    results = search_redis(redis_client, query_embedding, chunk_size, overlap)
                elif vector_db == "chroma":
                    results = search_chroma(chroma_client, query_embedding, chunk_size, overlap)
                elif vector_db == "qdrant":
                    results = search_qdrant(qdrant_client, query_embedding, chunk_size, overlap)

                for chunk, score in results:
                    write_to_file(OUTPUT_FILE, f" - Score: {score:.4f}, Text: {chunk[:100]}...")

if __name__ == "__main__":
    main()


