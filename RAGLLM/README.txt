Production Application of LLM-RAG Key Points due to level of functionality affecting Retrieving performance:

General Q&A : Use Vector Index retrievers + BM25 Store Indexes
Technical Documents: BM25 Store Indexes with Vector Index Retrievers
Research Papers : Recursive Retrievers
Long Documents: Auto Merging Retrievers
Large Document Set: Document Summary with Index Retriever and Vector Indexing


FAISS vs ChromDB Selection:

FAISS has multiple Indexing options convivence wise application exhibited while ChromaDB has HNSW. But FAISS has no meta data support, no distributed system support due to local single node approach. 

Hence to summarize use FAISS for Local only deployment and Library supported coding, use chromaDB for Fast Prototyping with meta data rich querying and Milvus for Distributed production scale.