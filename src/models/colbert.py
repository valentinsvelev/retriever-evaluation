import os
from tqdm import tqdm
from ragatouille import RAGPretrainedModel


class ColBERT:
    def __init__(self, dataset_name, corpus, queries):
        self.corpus = corpus
        self.queries = queries
        self.model = RAGPretrainedModel.from_pretrained(pretrained_model_name_or_path="colbert-ir/colbertv2.0", verbose=1, n_gpu=4)
        self.index_name = f"{dataset_name}_colbertv2_index"
        self.index_path = os.path.join("outputs/indexes", self.index_name)

    def index(self):
        if not os.path.exists(self.index_path):
            print(f"Index '{self.index_name}' not found. Building it now...")
            self.model.index(
                collection=list(self.corpus.values()),
                document_ids=list(self.corpus.keys()),
                index_name=self.index_name,
                max_document_length=512,
                split_documents=True,
                use_faiss=True,
                bsize=64
            )
        else:
            print(f"Found existing index at '{self.index_path}'. Skipping indexing.")

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