# DS4300--Practical-2

### The goal of this project was to create our own RAG machine and to find the best result through experimentation of different pipelines. Below are the variables we explored:

* Chunk Sizes: [200, 500, 1000]
* Chunk Overlap Sizes: [0, 50, 100]
* Vector Databases: ["redis", "chroma", "qdrant"]
* Embedding Models: ["thenlper/gte-base", "hkunlp/instructor-xl", "sentence-transformers/all-mpnet-base-v2"]
* LLMs: Ollama 2 & Mistral 7B


### To execute our experiments of the indexing pipeline and collection of important data,"main.py" is our main driver script. The file should run through multiple pipelines and output the results/metrics in a file named "pipeline_metrics.json." 
