import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import AutoModel, AutoTokenizer#, BitsAndBytesConfig
import pandas as pd
import numpy as np
import time
import json
import gzip
import os
import gc
from tqdm.auto import tqdm

from src.indexing.faiss_indexer import FaissIndexer
from src.data_handler import DataHandler
from src.configs.models import MODELS
from src.evaluator import Evaluator

import shutil
from src.misc import save_dense_embeddings, load_dense_embeddings, append_dense_embeddings_hdf5


DATASET_MAPPING = {
    "irds:msmarco-passage/dev/small": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2020/judged": "irds:msmarco-passage/dev/small",
}
ARCHIVE_ROOT = "/dataHDD1/masterthesis"


class NVEmbedEncoder:
    """
    Handles loading and encoding using the NV-Embed-v2 model.
    """
    def __init__(self, model_key: str, config: dict = None, device: str = "cuda"):
        """Initializes the NVEmbedEncoder."""
        self.model_key = model_key
        self.model_name = MODELS[model_key]["model_path"]
        self.config = config or {}
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        # self.bnb_cfg = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_compute_dtype=torch.bfloat16,
        #    bnb_4bit_use_double_quant=True
        # )
        self.max_length = 4096 #32768
        
        self._load_model()
    
    def _wrap_submodules_dataparallel(self):
        """See NV Embed v2 HF model card"""
        if not torch.cuda.is_available():
            return
        n = torch.cuda.device_count()
        if n < 2:
            return

        for k, m in list(self.model._modules.items()):
            self.model._modules[k] = DataParallel(m)

    def _load_model(self):
        """Loads the tokenizer and the NV-Embed-v2 model."""
        print(f"Loading NV-Embed model: {self.model_name}")

        # --- Step 1: Load tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )

        # --- Step 2: Load model ---
        self.model = AutoModel.from_pretrained(
            self.model_name,
            #load_in_4bit=True,
            #quantization_config=self.bnb_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            #device_map="auto",
            #attn_implementation="spda",
        )

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        
        # --- Step 3: Move to device and set to eval mode ---
        self.model.to(self.device)
        self.model.eval()
        self._wrap_submodules_dataparallel()

        print(f"NV-Embed model loaded successfully and moved to {self.device}.")


    def encode(
        self,
        texts,
        is_query=False,
        batch_size=16,           # global batch
        micro_batch_size=16,     # microbatch
        show_progress_bar=True,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        if micro_batch_size > batch_size:
            micro_batch_size = batch_size
        if batch_size % micro_batch_size != 0:
            # not strictly required, but keeps shapes predictable
            # we'll still handle the last uneven microbatch fine.
            pass

        instruction = self.config.get("query_instruction") if is_query else self.config.get("doc_instruction")
        instruction = instruction or ""

        all_embeddings = []
        desc = "Encoding Queries" if is_query else "Encoding Documents"

        # outer loop: global batches
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, disable=not show_progress_bar):
            big_batch_texts = texts[i : i + batch_size]

            # inner loop: microbatches
            micro_embs_cpu = []

            with torch.inference_mode():
                for j in range(0, len(big_batch_texts), micro_batch_size):
                    micro_texts = big_batch_texts[j : j + micro_batch_size]

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        embs = self.model.encode(
                            micro_texts,
                            instruction=instruction,
                            max_length=self.max_length,
                        )

                    if isinstance(embs, np.ndarray):
                        embs = torch.from_numpy(embs)

                    if isinstance(embs, torch.Tensor):
                        embs = embs.detach().cpu()
                        embs = F.normalize(embs, p=2, dim=1)

                    micro_embs_cpu.append(embs)

                    # drop refs ASAP
                    del embs, micro_texts

            # stitch microbatches back into the original global batch order
            big_batch_embs = torch.cat(micro_embs_cpu, dim=0)
            all_embeddings.append(big_batch_embs)

            del big_batch_embs, micro_embs_cpu, big_batch_texts

        if not all_embeddings:
            hidden_size = self.model.config.hidden_size
            return torch.empty(0, hidden_size)

        return torch.cat(all_embeddings, dim=0)



    def run(self, model: str, handler: DataHandler, ds: str, device: str, top_k: int = 1001, variant: str | None = None, save_report: bool = False, archive: bool = True):
        """
        Run NV-Embed in the same style as run.py.
        """
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

        # Skip if scores already exist
        scores_dir = f"outputs/scores/{self.model_key}"
        os.makedirs(scores_dir, exist_ok=True)
        scores_path = os.path.join(scores_dir, f"{dataset_label}.json")
        if os.path.exists(scores_path):
            print(f"Score found at {scores_path}. Skipping NV-Embed run...")
            # If you want, you can load & return here; for now just skip.
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

        print(f"--- Running Benchmark (NV-Embed) ---")
        print(f"Model: {self.model_key} | Checkpoint: {self.model_name} | Dataset: {dataset_label}")

        # ---------------------------
        # 1. Load pre-saved dataset
        # ---------------------------
        
        indexer = None # instantiate indexer outside of for loop
        all_doc_ids = []
        
        docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant, yield_batches=True)
        
        emb_path = f"outputs/embeddings/{self.model_key}/{corpus_label}.h5"

        if os.path.exists(emb_path):
            print(f"Reusing corpus embeddings from {emb_path}")
            embeddings, all_doc_ids = load_dense_embeddings_hdf5(emb_path)

            indexer = FaissIndexer(dimension=embeddings.shape[1])
            t = time.time()
            indexer.build(embeddings)
            timing["index_build_seconds"] += time.time() - t
        else:
            for d in docs_iter:
                doc_texts = [doc.get("text", "") or "" for doc in d]
                batch_doc_ids = [str(doc.get("doc_id", "") or "") for doc in d]
                all_doc_ids.extend(batch_doc_ids)
                
                # Compute document embeddings
                t = time.time()
                batch_embeddings = self.encode(
                    texts=doc_texts,
                    is_query=False,
                    batch_size=16,
                    show_progress_bar=True,
                )
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

                # Free memory
                del d, doc_texts, batch_doc_ids, batch_embeddings
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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
        query_embeddings = self.encode(
            texts=query_texts,
            is_query=True,
            batch_size=16,
            show_progress_bar=True,
        )
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
        
        
        
        # _, queries_iter, qrels_iter = handler.read(dataset_id, variant=variant)

        # query_ids, query_texts = [], []
        # for row in queries_iter:
        #     query_ids.append(str(row["query_id"]))
        #     query_texts.append(row["text"])

        # qrels = pd.DataFrame(list(qrels_iter))

        # docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)

        # doc_ids, doc_texts = [], []
        # for row in docs_iter:
        #     doc_ids.append(str(row["doc_id"]))
        #     doc_texts.append(row["text"])

        # timing["num_queries"] = len(query_ids)

        # del docs_iter, queries_iter
        # gc.collect()

        # # ---------------------------
        # # 2. Encode / cache corpus
        # # ---------------------------
        # out_dir = f"outputs/embeddings/{self.model_key}"
        # os.makedirs(out_dir, exist_ok=True)
        # emb_path = os.path.join(out_dir, f"{corpus_label}.npz")

        # if os.path.exists(emb_path):
        #     print(f"Loading cached corpus embeddings from {emb_path}")
        #     corpus_embeddings = load_dense_embeddings(emb_path)
        # else:
        #     print(f"No cached corpus embeddings at {emb_path}, encoding corpus with NV-Embed...")
        #     t0 = time.time()
        #     corpus_embeddings = self.encode(
        #         texts=doc_texts,
        #         is_query=False,
        #         batch_size=16,
        #         show_progress_bar=True,
        #     )
        #     timing["doc_encoding_seconds"] += time.time() - t0
            
        #     # Save if MS MARCO
        #     if dataset_id == "irds:msmarco-passage/dev/small":
        #         save_dense_embeddings(corpus_embeddings, emb_path)

        # # Free up RAM
        # del doc_texts, doc_titles
        # gc.collect()

        # # ---------------------------
        # # 3. Encode queries
        # # ---------------------------
        # t0 = time.time()
        # query_embeddings = self.encode(
        #     texts=query_texts,
        #     is_query=True,
        #     batch_size=16,
        #     show_progress_bar=True,
        # )
        # timing["query_encoding_seconds"] += time.time() - t0

        # # ---------------------------
        # # 4. FAISS index + search
        # # ---------------------------
        # indexer = FaissIndexer(dimension=corpus_embeddings.shape[1])
        # t0 = time.time()
        # indexer.build(corpus_embeddings)
        # timing["index_build_seconds"] += time.time() - t0

        # t0 = time.time()
        # scores, indices = indexer.search(query_embeddings, top_k=top_k)
        # timing["search_seconds"] += time.time() - t0

        # results = {
        #     qid: {doc_ids[idx]: float(score) for idx, score in zip(indices[i], scores[i])}
        #     for i, qid in enumerate(query_ids)
        # }

        # # ---------------------------
        # # 5. Save results (like run.py)
        # # ---------------------------
        # results_dir = f"outputs/results/{self.model_key}"
        # os.makedirs(results_dir, exist_ok=True)
        # results_path = os.path.join(results_dir, f"{dataset_label}.json")
        # with gzip.open(results_path, "wt", encoding="utf-8") as f:
        #     json.dump(results, f)
        # print(f"Saved search results to {results_path}")

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

        end_time = time.time()
        elapsed = end_time - start_time
        
        # Free up RAM
        if "jhu-clsp" not in dataset_id.lower():
            del results
            gc.collect()
            results = None

        # latency
        if timing["num_queries"] > 0 and timing["search_seconds"] > 0:
            avg_latency_ms = (timing["search_seconds"] / timing["num_queries"]) * 1000.0
        else:
            avg_latency_ms = None

        timing["avg_latency_ms"] = avg_latency_ms
        timing["evaluation_seconds"] = float(evaluation_seconds)
        timing["runtime_seconds"] = float(elapsed)

        print(f"This run took {float(elapsed)} seconds.")
        if avg_latency_ms is not None:
            print(f"Average query latency: {avg_latency_ms:.2f} ms/query "
                  f"over {timing['num_queries']} queries.")

        report = {
            "model_name": str(self.model_name),
            "dataset_id": dataset_id,
            "variant": variant,
            "metrics": metrics_agg,
            "summary_stats": summary_stats,
            "runtime_seconds": float(elapsed),
            "timing": timing,
        }

        if save_report:
            with open(scores_path, "w", encoding="utf-8") as f:
                json.dump(report, f)
            print(f"Saved metrics report to {scores_path}")

        # ---------------------------
        # 7. (Optional) archival like run.py
        # ---------------------------
        if archive:
            artefacts = []
    
            emb_path_for_arch = f"outputs/embeddings/{self.model_key}/{corpus_label}.h5"
            if os.path.exists(emb_path_for_arch):
                artefacts.append(emb_path_for_arch)
    
            result_file = f"outputs/results/{self.model_key}/{dataset_label}.json"
            if os.path.exists(result_file):
                artefacts.append(result_file)
    
            # If you want the same archival behavior:
            for folder in ["embeddings", "results"]:
                path = os.path.join(ARCHIVE_ROOT, f"outputs/{folder}/{self.model_key}")
                os.makedirs(path, exist_ok=True)
    
            for artefact in artefacts:
                dest = os.path.join(ARCHIVE_ROOT, artefact)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                print(f"[ARCHIVE] Copying → {artefact} → {dest}")
                if os.path.isdir(artefact):
                    shutil.copytree(artefact, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(artefact, dest)
                
                # Delete from NVME
                if corpus_id not in ["irds:msmarco-passage/dev", "irds:msmarco-passage/dev/small"]:
                    if os.path.isdir(artefact):
                        print(f"[CLEANUP] Removing folder from NVMe → {artefact}")
                        shutil.rmtree(artefact)
                    elif os.path.isfile(artefact):
                        print(f"[CLEANUP] Removing file from NVMe → {artefact}")
                        os.remove(artefact)

        # Clean up objects
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "metrics_agg": metrics_agg,
            "metrics_perq": metrics_perq,
            "summary_stats": summary_stats,
            "results": results,
            "runtime_seconds": float(elapsed),
            "variant": variant,
            "timing": timing,
        }
