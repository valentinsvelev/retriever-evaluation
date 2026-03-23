################################################################################
# faiss_indexer.py
#
# Description: Creates index (FAISS) for sparse models and performs top-k search.
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import numpy as np
import faiss


class FaissIndexer:
    """
    Handles FAISS indexing and searching.
    """
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    @staticmethod
    def _to_float32_numpy(x):
        # PyTorch tensor -> numpy
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()

        # Now it should be a numpy array
        x = np.asarray(x, dtype=np.float32)
        
        # FAISS expects float32
        return np.ascontiguousarray(x)

    def build(self, corpus_embeddings):
        print("Building FAISS index...")
        xb = self._to_float32_numpy(corpus_embeddings)
        self.index.add(xb)

    def search(self, query_embeddings, top_k):
        print("Searching index...")
        xq = self._to_float32_numpy(query_embeddings)
        return self.index.search(xq, k=top_k)
    