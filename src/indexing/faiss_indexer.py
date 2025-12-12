import faiss


class FaissIndexer:
    """Handles FAISS indexing and searching."""
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def build(self, corpus_embeddings):
        print("Building FAISS index...")
        self.index.add(corpus_embeddings.numpy().astype('float32'))

    def search(self, query_embeddings, top_k):
        print("Searching index...")
        return self.index.search(query_embeddings.numpy().astype('float32'), k=top_k)