# DS4300--Practical-2
# Overview
## This project utilizes several Python libraries and frameworks to perform document processing, semantic search, and vector-based similarity matching. The core of the project includes the following functionalities:
* Extracting text from PDF documents
*	Generating semantic embeddings for document content using Sentence Transformers
*	Storing and querying vectors in a database using Qdrant or ChromaDB
*	Efficiently calculating semantic similarity between documents or queries
* Optimizing resource usage using Redis and monitoring system stats with psutil

# Dependencies: 
* time
* json
* numpy
* redis
* pypdf
* sentence-transformers
* scipy
* tiktoken
* psutil
* chromadb
* qdrant-client

### The goal of this project was to create our own RAG machine and to find the best result through experimentation of different pipelines. Below are the variables we explored:

* Chunk Sizes: [200, 500, 1000]
* Chunk Overlap Sizes: [0, 50, 100]
* Vector Databases: ["redis", "chroma", "qdrant"]
* Embedding Models: ["thenlper/gte-base", "hkunlp/instructor-xl", "sentence-transformers/all-mpnet-base-v2"]
* LLMs: Ollama 2 & Mistral 7B


### To execute our experiments of the indexing pipelines and collection of important data,"main.py" is our main driver script. The file should run through multiple pipelines and output the results/metrics in a file named "pipeline_metrics.json." 

### Our "ingest.py" file includes the functions and packages for text extraction, chunking, generating embeddings, and storing and querying embeddings.

## Contributors
* Melina Yang
* Ravneet Kaur
* Angela Wu 


