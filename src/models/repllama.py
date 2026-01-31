################################################################################
# Title
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import time
import json
import gzip
import os
import gc
from tqdm import tqdm

from src.indexing.faiss_indexer import FaissIndexer
from src.data_handler import DataHandler
from src.configs.models import MODELS
from src.evaluator import Evaluator

import shutil
from src.misc import save_dense_embeddings, load_dense_embeddings_hdf5, append_dense_embeddings_hdf5, iter_dense_embeddings_hdf5, get_hdf5_embedding_dim


DATASET_MAPPING = {
    "irds:msmarco-passage/dev/small": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2020/judged": "irds:msmarco-passage/dev/small",
}
ARCHIVE_ROOT = "/dataHDD1/masterthesis"


class RepLLaMAEncoder:

    def __init__(self, model_key: str, config: dict = None, device: str = "cuda:0"):
        """Initializes the RepLLaMA."""
        self.model_key = model_key
        self.model_name = MODELS[model_key]["model_path"]
        self.config = config or {}
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Loads wrapper."""
        print(f"Loading RepLLaMA with its PEFT config...")
        peft_id = self.model_name
        peft_cfg = PeftConfig.from_pretrained(peft_id)
        base_id = peft_cfg.base_model_name_or_path
        
        base = AutoModel.from_pretrained(
            base_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "right"

        model = PeftModel.from_pretrained(base, peft_id)
        model = model.merge_and_unload()
        
        model.config.max_length = 512
        tokenizer.model_max_length = 512

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer


    def _apply_instruction(self, batch_texts, is_query: bool):
        instruction = self.config.get('query_instruction') if is_query else self.config.get('doc_instruction')
        if not instruction:
            return batch_texts
        if "{}" in instruction:
            return [instruction.format(text) for text in batch_texts]
        return [f'{instruction}{text}' for text in batch_texts]


    def _last_token_pool(self, last_hidden_state, attention_mask):
        idxs = attention_mask.sum(dim=1).long() - 1
        return last_hidden_state[torch.arange(last_hidden_state.size(0)), idxs]


    def encode(self, texts, is_query=False, batch_size=8):
        """Encodes a list of texts using the HF model directly"""
    
        # Build payload: same logic as before (instructions vs plain text)
        instruction = self.config.get("query_instruction") if is_query else self.config.get("doc_instruction")
        
        payload = [instruction.format(t) for t in texts]
    
        all_embs = []
        desc = "Encoding Queries" if is_query else "Encoding Documents"
    
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(payload), batch_size), desc=desc):
                batch_texts = payload[i : i + batch_size]
    
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )

                if torch.cuda.is_available():
                    first_device = next(self.model.parameters()).device
                    encoded = {k: v.to(first_device) for k, v in encoded.items()}
    
                outputs = self.model(**encoded)
                batch_embs = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                all_embs.append(batch_embs.cpu())

                del batch_embs

        embeddings = torch.cat(all_embs, dim=0)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    
    def run(self, model: str, handler: DataHandler, ds: str, device: str, top_k: int = 1001, variant: str | None = None, save_report: bool = False, archive: bool = True):
        # ---------------------------
        # 0. Dataset / corpus mapping
        # ---------------------------
        dataset_id = ds
        dataset_label = dataset_id.replace("/", "_").replace(":", "_")
        if variant:
            dataset_label = f"{dataset_label}_{variant}"

        corpus_id = DATASET_MAPPING.get(dataset_id, dataset_id)
        corpus_label = corpus_id.replace("/", "_").replace(":", "_")
        corpus_variant = variant if corpus_id == dataset_id else None

        scores_dir = f"outputs/scores/{self.model_key}"
        os.makedirs(scores_dir, exist_ok=True)
        scores_path = os.path.join(scores_dir, f"{dataset_label}.json")

        if os.path.exists(scores_path):
            print(f"Score found at {scores_path}. Skipping...")
            return None

        start_time = time.time()

        timing = {
            "doc_encoding_seconds": 0.0,
            "index_build_seconds": 0.0,
            "query_encoding_seconds": 0.0,
            "search_seconds": 0.0,
            "rerank_seconds": 0.0,
            "num_queries": 0,
        }

        print(f"--- Running Benchmark (RepLLaMA) ---")
        print(f"Model: {self.model_key} | Dataset: {dataset_label}")

        # ---------------------------
        # 1. Load dataset
        # ---------------------------
        indexer = None # instantiate indexer outside of for loop
        emb_path = f"outputs/embeddings/{self.model_key}/{corpus_label}.h5"

        if os.path.exists(emb_path):
            print(f"Reusing corpus embeddings from {emb_path}") 
            embeddings, all_doc_ids = load_dense_embeddings_hdf5(emb_path)
            
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant, yield_batches=True)

            all_doc_ids = []
            for d in docs_iter:
                all_doc_ids.extend([str(doc.get("doc_id", "") or "") for doc in d])
            
            indexer = FaissIndexer(dimension=embeddings.shape[1])
            t = time.time()
            indexer.build(embeddings)
            timing["index_build_seconds"] += time.time() - t

        else:
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant, yield_batches=True)
            all_doc_ids = []

            for d in docs_iter:
                doc_texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in d]
                batch_doc_ids = [str(doc.get("doc_id", "") or "") for doc in d]
                all_doc_ids.extend(batch_doc_ids)
                
                # Compute document embeddings
                t = time.time()
                batch_embeddings = self.encode(doc_texts, is_query=False, batch_size=8)
                timing["doc_encoding_seconds"] += time.time() - t
                
                # Save embeddings iteratively as json for MS MARCO
                out_dir = f"outputs/embeddings/{self.model_key}"
                os.makedirs(out_dir, exist_ok=True)
                emb_path = os.path.join(out_dir, f"{corpus_label}.h5")
                
                if dataset_id == "irds:msmarco-passage/dev/small":
                    append_dense_embeddings_hdf5(batch_embeddings, batch_doc_ids, emb_path)
                
                # Instantiate indexer
                if indexer is None:
                    indexer = FaissIndexer(dimension=batch_embeddings.shape[1])

                # Build index iteratively
                t = time.time()
                indexer.build(batch_embeddings)
                timing["index_build_seconds"] += time.time() - t
        
        # Load queries
        _, queries_iter, qrels_iter = handler.read(dataset_id, variant=variant)

        query_ids = []
        query_texts = []
        
        for q in queries_iter:
            query_ids.append(str(q.get("query_id", "") or ""))
            query_texts.append(q.get("text", "") or "")

        timing["num_queries"] = len(query_ids)

        # Compute query embeddings
        t0 = time.time()
        query_embeddings = self.encode(query_texts, is_query=True, batch_size=8)
        timing["query_encoding_seconds"] += time.time() - t0
        
        # Search index
        t0 = time.time()
        scores, indices = indexer.search(query_embeddings, top_k=top_k)
        timing["search_seconds"] += time.time() - t0

        del query_embeddings, indexer
        gc.collect()

        # Create results dict for evaluation
        results = {
            qid: {all_doc_ids[idx]: float(score) for idx, score in zip(indices[i], scores[i])}
            for i, qid in enumerate(query_ids)
        }
        
        # Save results as json
        results_dir = f"outputs/results/{self.model_key}"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{dataset_label}.json")
        with gzip.open(results_path, "wt", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"Saved search results to {results_path}")

        # ---------------------------
        # 6. Evaluate
        # ---------------------------
        qrels   = pd.DataFrame(list(qrels_iter))
        evaluator = Evaluator(dataset_id, skip_self_matches="auto")
        eval_start = time.time()
        metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels, results)
        evaluation_seconds = time.time() - eval_start

        print(f"\n--- Results for model {self.model_key} ---")
        print(pd.DataFrame([metrics_agg]))

        total_elapsed = time.time() - start_time

        # Free up RAM
        if "jhu-clsp" not in dataset_id.lower():
            del results
            gc.collect()
            results = None

        if timing["num_queries"] > 0:
            timing["avg_latency_ms"] = timing["search_seconds"] / timing["num_queries"] * 1000
        else:
            timing["avg_latency_ms"] = None

        timing["evaluation_seconds"] = float(evaluation_seconds)
        timing["runtime_seconds"] = float(total_elapsed)

        report = {
            "model_name": self.model_name,
            "dataset_id": dataset_id,
            "variant": variant,
            "metrics": metrics_agg,
            "summary_stats": summary_stats,
            "runtime_seconds": float(total_elapsed),
            "timing": timing,
        }

        if save_report:
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(report, f)
            print(f"Saved metrics report to {scores_path}")

        # ---------------------------
        # 7. Archival (same as run.py)
        # ---------------------------
        if archive:
            artefacts = []
    
            emb_path_arch = f"outputs/embeddings/{self.model_key}/{corpus_label}.h5"
            if os.path.exists(emb_path_arch):
                artefacts.append(emb_path_arch)
    
            result_file = f"outputs/results/{self.model_key}/{dataset_label}.json"
            if os.path.exists(result_file):
                artefacts.append(result_file)
    
            for folder in ["embeddings", "results"]:
                path = os.path.join(ARCHIVE_ROOT, f"outputs/{folder}/{self.model_key}")
                os.makedirs(path, exist_ok=True)
    
            for artefact in artefacts:
                dest = os.path.join(ARCHIVE_ROOT, artefact)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                print(f"[ARCHIVE] Copying → {artefact} → {dest}")
                shutil.copy2(artefact, dest)
    
                if corpus_id not in ["irds:msmarco-passage/dev", "irds:msmarco-passage/dev/small"]:
                    print(f"[CLEANUP] Removing from NVMe → {artefact}")
                    os.remove(artefact)

        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "metrics_agg": metrics_agg,
            "metrics_perq": metrics_perq,
            "summary_stats": summary_stats,
            "results": results,
            "runtime_seconds": float(total_elapsed),
            "variant": variant,
            "timing": timing,
        }
