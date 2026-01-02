import os
import torch
from tqdm import tqdm
from ragatouille import RAGPretrainedModel


class ColBERT:
    def __init__(self, corpus_label, corpus, queries):
        self.corpus = corpus
        self.queries = queries
        self.model = RAGPretrainedModel.from_pretrained(pretrained_model_name_or_path="colbert-ir/colbertv2.0", n_gpu=4) #1
        self.index_name = f"{corpus_label}_colbertv2_index"
        self.index_path = os.path.expanduser(os.path.join("~/.ragatouille/colbert/indexes", self.index_name))

    def index(self):
        print(f"Ensuring index '{self.index_name}' is ready...")
        self.model.index(
            collection=list(self.corpus.values()),
            document_ids=list(self.corpus.keys()),
            index_name=self.index_name,
            max_document_length=512,
            split_documents=True,
            use_faiss=True,
            bsize=32
        )

    def search(self):
        results = {}
        print(f"Searching {len(self.queries)} queries...")
        for qid, query_text in tqdm(self.queries.items()):
            hits = self.model.search(
                query=query_text,
                index_name=self.index_name,
                k=1001
            )
            results[qid] = {hit["document_id"]: float(hit["score"]) for hit in hits}
        print("Search complete.")

        return results