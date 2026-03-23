################################################################################
# sparta.py
#
# Description: Runs end-to-end retrieval experiments with SPARTA, including 
# document/query embedding generation, FAISS indexing, search, evaluation, 
# result/report saving, and optional artifact archiving.
#
# Adapted from https://github.com/nreimers/beir-sparta/blob/main/SPARTA.py
# and https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sparta.py
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

class SPARTA:
    def __init__(self, model_path: str, device: str = 'cpu', top_k: int = 512): # BEIR authors used topk=2000
        self.device = device
        self.top_k = top_k
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_input_embeddings = self.model.embeddings.word_embeddings.weight
        self.special_token_ids = torch.tensor(
            list(self.tokenizer.all_special_ids),
            device=self.device
        )
        self.special_token_id_set = set(self.tokenizer.all_special_ids)


    def _encode_batch(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=500).to(self.device) # BEIR used 500
        with torch.no_grad():
            doc_embs = self.model(**inputs).last_hidden_state
            scores = torch.einsum('bsh,vh->bvs', doc_embs, self.bert_input_embeddings) # equivalent to torch.matmul()
            max_scores = torch.max(scores, dim=-1).values
            transformed_scores = torch.log(torch.relu(max_scores) + 1) # dropped bias term for simplicity
            
            # Zero out special tokens
            transformed_scores[:, self.special_token_ids] = 0.0

            # Prune the vocabulary to the top_k most important terms
            top_k_scores, top_k_indices = torch.topk(transformed_scores, k=self.top_k, dim=-1)

        batch_dicts = []
        for i in range(top_k_scores.shape[0]):
            output_dict = {
                self.tokenizer.convert_ids_to_tokens(token_id.item()): score.item()
                for token_id, score in zip(top_k_indices[i], top_k_scores[i]) if score.item() > 0
            }
            batch_dicts.append(output_dict)

        return batch_dicts


    def encode(self, query: str) -> dict:
        ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]
        tf = defaultdict(int)

        for tid in ids:
            if tid in self.special_token_id_set:
                continue
            tok = self.tokenizer.convert_ids_to_tokens(tid)
            tf[tok] += 1

        return dict(tf)


    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int = 16) -> dict:
        sentences = [(doc["title"] + " " + doc["text"]).strip() if "title" in doc and len(doc["title"]) > 0 else doc["text"].strip() for doc in corpus]
        final_results = {}
        for i in tqdm(range(0, len(sentences), batch_size), desc='Encoding Corpus'):
            batch_sentences = sentences[i:i+batch_size]
            batch_ids = [corpus[j]['_id'] for j in range(i, i + len(batch_sentences))]
            batch_dicts = self._encode_batch(batch_sentences)
            for doc_id, doc_dict in zip(batch_ids, batch_dicts):
                final_results[doc_id] = doc_dict
        return final_results