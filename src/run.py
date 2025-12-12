import pandas as pd
import numpy as np
import torch
import json
import gzip
import os
import time
import gc
import shutil
from tqdm import tqdm
import pyarrow.parquet as pq
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher

from src.configs.models import MODELS
from src.encoders.dense_encoder import DenseEncoder, generate_docs_for_query_expansion
from src.encoders.sparse_encoder import SparseEncoder
from src.indexing.faiss_indexer import FaissIndexer
from src.indexing.lucene_indexer import LuceneIndexer
from src.evaluator import Evaluator
from src.data_handler import DataHandler
from src.models.colbert import ColBERT
from src.misc import prepare_pyserini_corpus, save_dense_embeddings, load_dense_embeddings


DATASET_MAPPING = {
    "irds:msmarco-passage/dev/small": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2020/judged": "irds:msmarco-passage/dev/small",
}

def run(
    model_key: str, 
    handler: DataHandler,
    ds: str, device: str, 
    top_k: int = 1001,
    variant = None,
    save_report: bool = False,
    archive: bool = True
    ):

    # For all datasets
    dataset_id = ds
    dataset_label = dataset_id.replace("/", "_").replace(":", "_")
    if variant:
        dataset_label = f"{dataset_label}_{variant}"
    
    # For MS MARCO datasets
    corpus_id = DATASET_MAPPING.get(dataset_id, dataset_id)
    corpus_label = corpus_id.replace("/", "_").replace(":", "_")

    corpus_variant = variant if corpus_id == dataset_id else None

    # Run retrieval only if a result doesnt exist
    if not os.path.exists(f"outputs/scores/{model_key}/{dataset_label}.json"):
        start_time = time.time()

        timing = {
            "doc_encoding_seconds": 0.0,   # corpus encoding / expansion
            "index_build_seconds": 0.0,    # FAISS / Lucene / ColBERT index build
            "query_encoding_seconds": 0.0, # dense query encoding
            "search_seconds": 0.0,         # search over index (incl. sparse query proc)
            "rerank_seconds": 0.0,         # TART reranking only
            "num_queries": 0,              # filled after we know len(query_ids)
        }

        # ---------------------------
        # 0. Model config/label
        # ---------------------------
        if model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct"]:
            if model_key != "bm25":
                model_name = MODELS[model_key]["model_path"]
            else:
                model_name = model_key
        elif model_key == "splade":
            model_name = MODELS[model_key]["model_path"]
        elif model_key == "colbert":
            model_name = "colbert"
        else:
            if model_key not in MODELS:
                raise ValueError(f"Model key '{model_key}' not found.")

            if "dpr" in model_key or "dragon" in model_key:
                model_name = {
                    "question": MODELS[model_key]["model_path_q"],
                    "context":  MODELS[model_key]["model_path_ctx"],
                }
            else:
                model_name = MODELS[model_key]["model_path"]

            config = MODELS[model_key]

        print(f"--- Running Benchmark ---")
        print(f"Model: {model_key} | Checkpoint: {model_name} | Dataset: {dataset_label}")

        # ---------------------------
        # 1. Load pre-saved dataset
        # ---------------------------

        # Queries & qrels always from the evaluation dataset
        _, queries_iter, qrels_iter = handler.read(dataset_id, variant=variant)
        queries = pd.DataFrame(list(queries_iter))
        qrels   = pd.DataFrame(list(qrels_iter))

        # Docs from the underlying corpus, with corpus_variant logic
        docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
        docs = pd.DataFrame(list(docs_iter))

        doc_ids    = docs["doc_id"].astype(str).tolist()
        doc_texts  = docs["text"].tolist()
        doc_titles = docs["title"].tolist() if "title" in docs.columns else None

        query_ids   = queries["query_id"].astype(str).tolist()
        query_texts = queries["text"].tolist()

        timing["num_queries"] = len(query_ids)

        # Free up memory
        del docs, queries, docs_iter, queries_iter
        gc.collect()

        # ---------------------------
        # 2–3. Encode/Index/Search
        # ---------------------------

        results = None

        # ---------- Sparse retrievers ----------
        if model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]:
            corpus_dir = f"data/corpora/{corpus_label}"

            if os.path.exists(corpus_dir):
                print(f"Reusing corpus directory {corpus_dir}")
            else:
                print(f"Building corpus directory {corpus_dir}")
                prepare_pyserini_corpus(docs, corpus_dir, corpus_label)

            model = SparseEncoder(model_name=model_name, model_key=model_key, device=device)
            index_input = corpus_dir

            # Doc2Query expansion = offline doc "encoding" cost
            if model_key == "doc2query":
                t0 = time.time()
                index_input = model.build_d2q_docs(
                    docs_df=docs,
                    out_dir=f"data/augmented/d2q/{dataset_label}-d2q-expanded",
                    model_ckpt=model_name,
                    queries_per_doc=3,
                    max_input_len=512,
                    max_query_len=64,
                    batch_size=64,
                    top_k=10,
                    do_sample=True
                )
                timing["doc_encoding_seconds"] += time.time() - t0

            # Learned sparse encoders: doc encoding
            if model_key in ["unicoil", "sparta", "deepct", "splade"]:  # all but BM25 and doc2query
                enc_dir = f"outputs/encodings/{model_key}/{corpus_label}"
                if os.path.exists(enc_dir):
                    print(f"Reusing existing encodings at {enc_dir}")
                    index_input = enc_dir
                else:
                    print(f"Encoding corpus for {model_key} → {enc_dir}")
                    t0 = time.time()
                    index_input = model.encode(corpus_dir=index_input, encoding_dir=enc_dir)
                    timing["doc_encoding_seconds"] += time.time() - t0

            index_dir = f"outputs/indexes/{model_key}/{corpus_label}"
            indexer = LuceneIndexer(
                model_name=model_key,
                dataset_name=corpus_label,
                index_dir=index_dir,
            )

            if os.path.exists(index_dir):
                print(f"Reusing existing Lucene index at {index_dir}")
            else:
                print(f"Building Lucene index at {index_dir}")
                t0 = time.time()
                indexer.build(corpus_dir=index_input)
                timing["index_build_seconds"] += time.time() - t0

            print(f"\n--- Searching with model {model_key} ---")
            if model_key in ["bm25", "doc2query", "deepct"]:
                print(indexer.index_dir)
                searcher = LuceneSearcher(indexer.index_dir)
                searcher.set_bm25(k1=0.9, b=0.4)
            else:
                searcher = LuceneImpactSearcher(indexer.index_dir, model.query_encoder)

            results = {}
            search_start = time.time()
            for qid, qtext in zip(query_ids, query_texts):
                hits = searcher.search(qtext, k=top_k)
                results[qid] = {hit.docid: float(hit.score) for hit in hits}
            timing["search_seconds"] += time.time() - search_start

        # ---------- ColBERT ----------
        elif model_key == "colbert":
            corpus_dict  = dict(zip(doc_ids, doc_texts))
            queries_dict = {qid: f"query: {txt}" for qid, txt in zip(query_ids, query_texts)}
            colbert = ColBERT(dataset_name=dataset_label, corpus=corpus_dict, queries=queries_dict)

            # Free up RAM
            del doc_texts, doc_titles
            gc.collect()

            # ColBERT indexing (includes doc encoding + index build)
            t0 = time.time()
            colbert.index()
            timing["index_build_seconds"] += time.time() - t0

            # ColBERT search (includes query encoding + search)
            t0 = time.time()
            results = colbert.search()
            timing["search_seconds"] += time.time() - t0

        # ---------- HyDE (Contriever as retriever) ----------
        elif model_key == "hyde":
            hyde_queries = generate_docs_for_query_expansion(
                query_ids,
                query_texts,
                model_name,
                device,
                f"data/augmented/{dataset_label}-hyde-augmented.json"
            )

            encoder = DenseEncoder(MODELS["contriever"]["model_path"], MODELS["contriever"], device)

            out_dir = f"outputs/embeddings/contriever"
            os.makedirs(out_dir, exist_ok=True)
            emb_path = os.path.join(out_dir, f"{corpus_label}.npz")

            # --- Corpus encoding ---
            if os.path.exists(emb_path):
                print(f"Loading cached corpus embeddings from {emb_path}")
                corpus_embeddings = load_dense_embeddings(emb_path)
            else:
                print(f"No cached corpus embeddings at {emb_path}, encoding corpus...")
                t0 = time.time()
                corpus_embeddings = encoder.encode(
                    texts=doc_texts,
                    titles=doc_titles,
                    is_query=False,
                )
                timing["doc_encoding_seconds"] += time.time() - t0
                
                # Save if MS MARCO
                if dataset_id == "irds:msmarco-passage/dev/small":
                    save_dense_embeddings(corpus_embeddings, emb_path)
                
            # --- Query encoding: original + HyDE, then average ---
            t0 = time.time()
            # Original contriever query embedding
            q_orig = encoder.encode(texts=query_texts, is_query=True)

            # HyDE docs encoded in "doc mode" (matches corpus space)
            q_hyde = encoder.encode(texts=hyde_queries, is_query=False)

            # Equation (8) with N = 1  → average
            query_embeddings = 0.5 * (q_orig + q_hyde)
            timing["query_encoding_seconds"] += time.time() - t0

            # --- FAISS search ---
            indexer = FaissIndexer(dimension=corpus_embeddings.shape[1])
            t0 = time.time()
            indexer.build(corpus_embeddings)
            timing["index_build_seconds"] += time.time() - t0

            t0 = time.time()
            scores, indices = indexer.search(query_embeddings, top_k=top_k)
            timing["search_seconds"] += time.time() - t0

            results = {
                qid: {doc_ids[idx]: float(score) for idx, score in zip(indices[i], scores[i])}
                for i, qid in enumerate(query_ids)
            }

        # ---------- TART (Contriever retriever + TART reranker) ----------
        elif model_key == "tart":
            print(f"--- Running TART Reranker Pipeline with Contriever ---")
            
            # --- 1. Stage 1: Dense Retrieval with Contriever ---
            retriever_encoder = DenseEncoder(MODELS["contriever"]["model_path"], MODELS["contriever"], device)

            # Corpus Embeddings Path
            out_dir = f"outputs/embeddings/contriever"
            os.makedirs(out_dir, exist_ok=True)
            emb_path = os.path.join(out_dir, f"{corpus_label}.npz")
            
            #emb_path = "outputs/embeddings/contriever/irds_msmarco-passage_dev.npz"

            # Encode Corpus (or Load Cache)
            if os.path.exists(emb_path):
                print(f"[TART] Loading cached corpus embeddings from {emb_path}")
                corpus_embeddings = load_dense_embeddings(emb_path)
            else:
                print(f"[TART] Encoding corpus with Contriever...")
                t0 = time.time()
                corpus_embeddings = retriever_encoder.encode(texts=doc_texts, titles=doc_titles, is_query=False)
                timing["doc_encoding_seconds"] += time.time() - t0
                
                # Save if MS MARCO
                if dataset_id == "irds:msmarco-passage/dev/small":
                    save_dense_embeddings(corpus_embeddings, emb_path)

            # Build FAISS Index
            indexer = FaissIndexer(dimension=corpus_embeddings.shape[1])
            t0 = time.time()
            indexer.build(corpus_embeddings)
            timing["index_build_seconds"] += time.time() - t0

            # Encode Queries (retriever)
            t0 = time.time()
            query_embeddings = retriever_encoder.encode(texts=query_texts, is_query=True)
            timing["query_encoding_seconds"] += time.time() - t0
            
            # Retrieve Candidates
            #rerank_depth = min(top_k, 100) 
            t0 = time.time()
            scores_1st, indices_1st = indexer.search(query_embeddings, top_k=top_k)
            timing["search_seconds"] += time.time() - t0

            # Clean up retriever to free VRAM for TART
            del retriever_encoder
            del corpus_embeddings
            del indexer
            torch.cuda.empty_cache()

            # --- 2. Stage 2: Reranking with TART ---
            print(f"[TART] Stage 2: Reranking with {model_name}")
            tart_encoder = DenseEncoder(model_name, config, device)

            # Map for O(1) text access
            doc_map = dict(zip(doc_ids, doc_texts))
            instruction = config.get("instruction", "Find the passage that answers the given query.")
            results = {}

            rerank_start = time.time()
            for i, (qid, qtext) in tqdm(enumerate(zip(query_ids, query_texts)), total=len(query_ids), desc="TART Reranking"):
                # Get candidates from Stage 1
                cand_indices = indices_1st[i]
                cand_ids = [doc_ids[idx] for idx in cand_indices]
                cand_texts = [doc_map.get(did, "") for did in cand_ids]

                # Rerank
                scores_list = tart_encoder.rerank(
                    queries=[qtext], 
                    docs_per_query=[cand_texts], 
                    instruction=instruction,
                    batch_size=16
                )
                
                # Store Results
                results[qid] = {did: float(s) for did, s in zip(cand_ids, scores_list[0])}
            timing["rerank_seconds"] += time.time() - rerank_start

        # ---------- Generic dense retrieval ----------
        else:
            encoder = DenseEncoder(model_name, config, device)

            out_dir = f"outputs/embeddings/{model_key}"
            os.makedirs(out_dir, exist_ok=True)
            emb_path = os.path.join(out_dir, f"{corpus_label}.npz")
            if os.path.exists(emb_path):
                print(f"Loading cached corpus embeddings from {emb_path}")
                corpus_embeddings = load_dense_embeddings(emb_path)
            else:
                print(f"No cached corpus embeddings at {emb_path}, encoding corpus...")
                t0 = time.time()
                corpus_embeddings = encoder.encode(texts=doc_texts, titles=doc_titles, is_query=False)
                timing["doc_encoding_seconds"] += time.time() - t0
                
                # Save if Contriever or MS MARCO
                if model_key == "contriever" or dataset_id == "irds:msmarco-passage/dev/small":
                    save_dense_embeddings(corpus_embeddings, emb_path)
            
            # Free up RAM
            del doc_texts, doc_titles
            gc.collect()

            # Queries are dataset-specific
            t0 = time.time()
            query_embeddings = encoder.encode(texts=query_texts, is_query=True)
            timing["query_encoding_seconds"] += time.time() - t0

            indexer = FaissIndexer(dimension=corpus_embeddings.shape[1])
            t0 = time.time()
            indexer.build(corpus_embeddings)
            timing["index_build_seconds"] += time.time() - t0

            t0 = time.time()
            scores, indices = indexer.search(query_embeddings, top_k=top_k)
            timing["search_seconds"] += time.time() - t0

            results = {
                qid: {doc_ids[idx]: float(score) for idx, score in zip(indices[i], scores[i])}
                for i, qid in enumerate(query_ids)
            }

        # ---------------------------
        # 4. Save & Evaluate
        # ---------------------------
        out_dir = f"outputs/results/{model_key}"
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"{dataset_label}.json")
        with gzip.open(results_path, 'wt', encoding="utf-8") as f:
            json.dump(results, f)
        print(f"Saved search results to {results_path}")

        # Free up RAM
        del results
        gc.collect()

        evaluator = Evaluator(dataset_id, skip_self_matches="auto")

        eval_start = time.time()
        metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels, results)
        evaluation_seconds = time.time() - eval_start

        print(f"\n--- Results for model {model_key} ---")
        print(pd.DataFrame([metrics_agg]))

        end_time = time.time()
        elapsed = end_time - start_time

        # Final timing post-processing
        if timing["num_queries"] > 0 and timing["search_seconds"] > 0:
            avg_latency_ms = (timing["search_seconds"] / timing["num_queries"]) * 1000.0
        else:
            avg_latency_ms = None

        timing["avg_latency_ms"] = avg_latency_ms
        timing["evaluation_seconds"] = float(evaluation_seconds)
        timing["runtime_seconds"] = float(elapsed)  # whole pipeline up to end of eval

        print(f"This run took {float(elapsed)} seconds.")
        if avg_latency_ms is not None:
            print(f"Average query latency: {avg_latency_ms:.2f} ms/query "
                  f"over {timing['num_queries']} queries.")

        report = {
            "model_name": str(model_name),
            "dataset_id": dataset_id,
            "variant": variant,
            "metrics": metrics_agg,
            "summary_stats": summary_stats,
            "runtime_seconds": float(elapsed),
            "timing": timing,
        }

        if save_report:
            out_dir = f"outputs/scores/{model_key}"
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(out_dir, f"{dataset_label}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f)
            print(f"Saved metrics report to {report_path}")

        # ---------------------------
        # 5. Clean up
        # ---------------------------
        if model_key == "colbert":
            del colbert
        elif model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]:
            pass
        elif model_key == "tart":
            del tart_encoder
        else:
            del encoder, indexer, corpus_embeddings, query_embeddings
        del evaluator
        
        # Free up RAM
        gc.collect()

        # Free up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------
        # 6. Archival (move files to HDD)
        # ---------------------------
        if archive:
            ARCHIVE_ROOT = "/dataHDD1/masterthesis"

            # Create paths
            artefacts = []

            ## Sparse retrievers
            if model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]:
                enc_dir    = f"outputs/encodings/{model_key}/{corpus_label}"
                index_dir  = f"outputs/indexes/{model_key}/{corpus_label}"
                
                if os.path.exists(enc_dir):    artefacts.append(enc_dir)
                if os.path.exists(index_dir):  artefacts.append(index_dir)

            ## Dense retrievers (generic + hyde)
            if model_key in ["hyde"] or (model_key not in ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade", "colbert"]):
                emb_path = f"outputs/embeddings/{model_key}/{corpus_label}.npz"
                if os.path.exists(emb_path): artefacts.append(emb_path)

            ## ColBERT
            if model_key == "colbert":
                colbert_index_dir = f"colbert/indexes/{dataset_label}"
                if os.path.exists(colbert_index_dir): artefacts.append(colbert_index_dir)

            ## Results (common)
            result_file = f"outputs/results/{model_key}/{dataset_label}.json"
            if os.path.exists(result_file): artefacts.append(result_file)

            # Copy to HDD
            for folder in ["embeddings", "encodings", "indexes", "results"]:
                path = os.path.join(ARCHIVE_ROOT, f"outputs/{folder}/{model_key}")
                os.makedirs(path, exist_ok=True)

            for artefact in artefacts:
                # Copy to HDD
                if os.path.isdir(artefact):
                    # Directory → copy tree
                    dest_dir = os.path.join(ARCHIVE_ROOT, artefact)
                    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                    print(f"[ARCHIVE] Copying folder → {artefact} → {dest_dir}")
                    shutil.copytree(artefact, dest_dir, dirs_exist_ok=True)

                elif os.path.isfile(artefact):
                    # File → copy into parent directory in HDD
                    dest_file = os.path.join(ARCHIVE_ROOT, artefact)
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    print(f"[ARCHIVE] Copying file → {artefact} → {dest_file}")
                    shutil.copy2(artefact, dest_file)

                else:
                    print(f"[ARCHIVE] Warning: Artefact not found: {artefact}")
                    continue
                
                # Delete from NVME
                if corpus_id not in ["irds:msmarco-passage/dev", "irds:msmarco-passage/dev/small"]:
                    if os.path.isdir(artefact):
                        print(f"[CLEANUP] Removing folder from NVMe → {artefact}")
                        shutil.rmtree(artefact)
                    elif os.path.isfile(artefact):
                        print(f"[CLEANUP] Removing file from NVMe → {artefact}")
                        os.remove(artefact)
        
        # ---------------------------
        # 6b. Cleanup sparse artefacts when not archiving
        # ---------------------------
        if not archive:
            # Only consider sparse models
            if model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]:
                enc_dir   = f"outputs/encodings/{model_key}/{corpus_label}"
                index_dir = f"outputs/indexes/{model_key}/{corpus_label}"
                corpus_dir = f"data/corpora/{corpus_label}"

                # Keep MS MARCO variants, delete everything else
                if corpus_id not in ["irds:msmarco-passage/dev", "irds:msmarco-passage/dev/small"]:
                    for artefact in [enc_dir, index_dir, corpus_dir]:
                        if os.path.isdir(artefact):
                            print(f"[CLEANUP] Removing sparse directory from NVMe → {artefact}")
                            shutil.rmtree(artefact)
                        elif os.path.isfile(artefact):
                            print(f"[CLEANUP] Removing sparse file from NVMe → {artefact}")
                            os.remove(artefact)


        # Return everything that was computed
        return {
            "metrics_agg": metrics_agg,
            "metrics_perq": metrics_perq,
            "summary_stats": summary_stats,
            "results": results,
            "runtime_seconds": float(elapsed),
            "variant": variant,
            "timing": timing,
        }

    else:
        print(f"Score found at outputs/scores/{model_key}/{dataset_label}.json. Skipping...")
