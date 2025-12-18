import os
import json
import gzip
import time
import gc
import shutil
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher

from src.configs.models import MODELS
from src.encoders.dense_encoder import DenseEncoder, generate_docs_for_query_expansion
from src.encoders.sparse_encoder import SparseEncoder
from src.indexing.faiss_indexer import FaissIndexer
from src.indexing.lucene_indexer import LuceneIndexer
from src.evaluator import Evaluator
from src.data_handler import DataHandler
from src.models.colbert import ColBERT
from src.misc import save_dense_embeddings, load_dense_embeddings


DATASET_MAPPING = {
    "irds:msmarco-passage/dev/small": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2020/judged": "irds:msmarco-passage/dev/small",
}


# ---------------------------
# Helpers: streaming corpus creation (no pandas)
# ---------------------------
def ensure_pyserini_corpus_jsonl(
    docs_iter: Iterable[Dict[str, Any]],
    corpus_dir: str,
    *,
    concat_title: bool = True,
) -> str:
    """
    Creates data/corpora/<corpus_label>/corpus.jsonl in a streaming way.
    Format expected by Pyserini: {"id": "...", "contents": "..."} per line.
    Returns output path.
    """
    os.makedirs(corpus_dir, exist_ok=True)
    output_path = os.path.join(corpus_dir, "corpus.jsonl")
    if os.path.exists(output_path):
        return output_path

    with open(output_path, "w", encoding="utf-8") as f:
        for d in tqdm(docs_iter, desc=f"Writing {output_path}"):
            doc_id = str(d.get("doc_id", ""))
            text = d.get("text", "") or ""
            title = d.get("title", "") or ""

            if concat_title and title.strip():
                contents = (title.strip() + " " + str(text).strip()).strip()
            else:
                contents = str(text).strip()

            rec = {"id": doc_id, "contents": contents}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return output_path


def stream_encode_corpus_dense(
    docs_iter: Iterable[Dict[str, Any]],
    encoder: DenseEncoder,
    *,
    is_query: bool = False,
    batch_size: int = 2048,
    use_titles: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Stream documents from docs_iter, encode in batches, return:
      - corpus_embeddings: (N, d) numpy array
      - doc_ids: list of doc_id strings aligned with embeddings
    This avoids ever holding all doc_texts in memory.
    """
    all_ids: List[str] = []
    emb_chunks: List[np.ndarray] = []

    batch_texts: List[str] = []
    batch_titles: List[str] = []
    batch_ids: List[str] = []

    for d in tqdm(docs_iter, desc="Streaming corpus encoding"):
        batch_ids.append(str(d.get("doc_id", "")))
        batch_texts.append(d.get("text", "") or "")

        if use_titles:
            batch_titles.append(d.get("title", "") or "")

        if len(batch_ids) >= batch_size:
            titles_arg = batch_titles if use_titles else None
            embs = encoder.encode(texts=batch_texts, titles=titles_arg, is_query=is_query)
            emb_chunks.append(embs)
            all_ids.extend(batch_ids)

            batch_texts.clear()
            batch_titles.clear()
            batch_ids.clear()

            gc.collect()

    # last partial batch
    if batch_ids:
        titles_arg = batch_titles if use_titles else None
        embs = encoder.encode(texts=batch_texts, titles=titles_arg, is_query=is_query)
        emb_chunks.append(embs)
        all_ids.extend(batch_ids)

    corpus_embeddings = np.vstack(emb_chunks) if len(emb_chunks) > 1 else emb_chunks[0]
    return corpus_embeddings, all_ids


def run(
    model_key: str,
    handler: DataHandler,
    ds: str,
    device: str,
    top_k: int = 1001,
    variant=None,
    save_report: bool = False,
    archive: bool = True,
):
    dataset_id = ds
    dataset_label = dataset_id.replace("/", "_").replace(":", "_")
    if variant:
        dataset_label = f"{dataset_label}_{variant}"

    corpus_id = DATASET_MAPPING.get(dataset_id, dataset_id)
    corpus_label = corpus_id.replace("/", "_").replace(":", "_")
    corpus_variant = variant if corpus_id == dataset_id else None

    score_path = f"outputs/scores/{model_key}/{dataset_label}.json"
    if os.path.exists(score_path):
        print(f"Score found at {score_path}. Skipping...")
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

    #############################
    # ---------------------------
    # 0. Model config/label
    # ---------------------------
    #############################
    if model_key in ["bm25", "doc2query", "unicoil", "sparta", "deepct"]:
        model_name = MODELS[model_key]["model_path"] if model_key != "bm25" else "bm25"
        config = None
    elif model_key == "splade":
        model_name = MODELS[model_key]["model_path"]
        config = None
    elif model_key == "colbert":
        model_name = "colbert"
        config = None
    else:
        if model_key not in MODELS:
            raise ValueError(f"Model key '{model_key}' not found.")
        if "dpr" in model_key or "dragon" in model_key:
            model_name = {
                "question": MODELS[model_key]["model_path_q"],
                "context": MODELS[model_key]["model_path_ctx"],
            }
        else:
            model_name = MODELS[model_key]["model_path"]
        config = MODELS[model_key]

    print(f"--- Running Benchmark ---")
    print(f"Model: {model_key} | Checkpoint: {model_name} | Dataset: {dataset_label}")

    #############################
    # ---------------------------
    # 1. Load queries + qrels (small)
    # ---------------------------
    #############################
    _, queries_iter, qrels_iter = handler.read(dataset_id, variant=variant)
    queries = pd.DataFrame(list(queries_iter))
    qrels = pd.DataFrame(list(qrels_iter))

    query_ids = queries["query_id"].astype(str).tolist()
    query_texts = queries["text"].tolist()
    timing["num_queries"] = len(query_ids)

    # Decide whether we need the corpus in memory:
    sparse_keys = ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]
    needs_dense_corpus = model_key not in sparse_keys and model_key != "colbert"
    needs_colbert_corpus = (model_key == "colbert")
    needs_tart = (model_key == "tart")
    needs_hyde = (model_key == "hyde")

    corpus_dir = f"data/corpora/{corpus_label}"
    corpus_jsonl = os.path.join(corpus_dir, "corpus.jsonl")

    results = None

    #############################
    # ---------------------------
    # 2–3. Sparse retrievers
    # ---------------------------
    #############################
    if model_key in sparse_keys:
        # If corpus.jsonl exists and we're not doc2query, do NOT load docs at all.
        need_build_corpus = (not os.path.exists(corpus_jsonl))
        need_docs_df = (model_key == "doc2query")  # doc2query requires docs_df in your encoder

        docs_df = None
        if need_build_corpus and not need_docs_df:
            print(f"Building Pyserini corpus JSONL (streaming) at {corpus_jsonl}")
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
            ensure_pyserini_corpus_jsonl(docs_iter, corpus_dir, concat_title=True)

        if need_docs_df:
            print("doc2query selected → loading docs into DataFrame (this is inherently RAM-heavy).")
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
            docs_list = list(docs_iter)  # unavoidable with current build_d2q_docs signature
            docs_df = pd.DataFrame(docs_list)
            del docs_list
            gc.collect()

            if not os.path.exists(corpus_jsonl):
                # build corpus JSONL from DF (cheaper than re-reading)
                os.makedirs(corpus_dir, exist_ok=True)
                with open(corpus_jsonl, "w", encoding="utf-8") as f:
                    for _, row in tqdm(docs_df.iterrows(), total=len(docs_df), desc="Writing corpus.jsonl"):
                        doc_id = str(row.get("doc_id", ""))
                        text = row.get("text", "") or ""
                        title = row.get("title", "") or ""
                        contents = (str(title).strip() + " " + str(text).strip()).strip() if str(title).strip() else str(text).strip()
                        f.write(json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False) + "\n")

        model = SparseEncoder(model_name=model_name, model_key=model_key, device=device)
        index_input = corpus_dir

        # doc2query expansion
        if model_key == "doc2query":
            t0 = time.time()
            index_input = model.build_d2q_docs(
                docs_df=docs_df,
                out_dir=f"data/augmented/d2q/{dataset_label}-d2q-expanded",
                model_ckpt=model_name,
                queries_per_doc=3,
                max_input_len=512,
                max_query_len=64,
                batch_size=64,
                top_k=10,
                do_sample=True,
            )
            timing["doc_encoding_seconds"] += time.time() - t0

        # Now we can delete docs_df if it exists
        if docs_df is not None:
            del docs_df
            gc.collect()

        # learned sparse encoders: doc encoding
        if model_key in ["unicoil", "sparta", "deepct", "splade"]:
            enc_dir = f"outputs/encodings/{model_key}/{corpus_label}"
            if os.path.exists(enc_dir):
                print(f"Reusing existing encodings at {enc_dir}")
                index_input = enc_dir
            else:
                print(f"Encoding corpus for {model_key} → {enc_dir}")
                t0 = time.time()
                index_input = model.encode(corpus_dir=index_input, encoding_dir=enc_dir)
                timing["doc_encoding_seconds"] += time.time() - t0

        # build / reuse Lucene index
        index_dir = f"outputs/indexes/{model_key}/{corpus_label}"
        indexer = LuceneIndexer(model_name=model_key, dataset_name=corpus_label, index_dir=index_dir)

        if os.path.exists(index_dir):
            print(f"Reusing existing Lucene index at {index_dir}")
        else:
            print(f"Building Lucene index at {index_dir}")
            t0 = time.time()
            indexer.build(corpus_dir=index_input)
            timing["index_build_seconds"] += time.time() - t0

        print(f"\n--- Searching with model {model_key} ---")
        if model_key in ["bm25", "doc2query", "deepct"]:
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

        # cleanup sparse objects
        del searcher, indexer, model
        gc.collect()

    #############################
    # ---------------------------
    # 2–3. ColBERT
    # ---------------------------
    #############################
    elif model_key == "colbert":
        docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)

        # NOTE: ColBERT needs the corpus in a dict; this will always cost RAM.
        # (You can only truly fix this by changing ColBERT wrapper to accept paths/streaming.)
        corpus_dict = {}
        for d in tqdm(docs_iter, desc="Loading corpus for ColBERT"):
            corpus_dict[str(d.get("doc_id", ""))] = d.get("text", "") or ""

        queries_dict = {qid: f"query: {txt}" for qid, txt in zip(query_ids, query_texts)}
        colbert = ColBERT(dataset_name=dataset_label, corpus=corpus_dict, queries=queries_dict)

        # free the big dict reference only if ColBERT copies internally; otherwise keep.
        # We'll assume ColBERT stores it; keeping corpus_dict reference isn't needed.
        del corpus_dict
        gc.collect()

        t0 = time.time()
        colbert.index()
        timing["index_build_seconds"] += time.time() - t0

        t0 = time.time()
        results = colbert.search()
        timing["search_seconds"] += time.time() - t0

        del colbert
        gc.collect()

    #############################
    # ---------------------------
    # 2–3. HyDE (Contriever retriever)
    # ---------------------------
    #############################
    elif model_key == "hyde":
        hyde_queries = generate_docs_for_query_expansion(
            query_ids,
            query_texts,
            model_name,
            device,
            f"data/augmented/{dataset_label}-hyde-augmented.json",
        )

        encoder = DenseEncoder(MODELS["contriever"]["model_path"], MODELS["contriever"], device)

        out_dir = "outputs/embeddings/contriever"
        os.makedirs(out_dir, exist_ok=True)
        emb_path = os.path.join(out_dir, f"{corpus_label}.npz")

        # corpus embeddings
        if os.path.exists(emb_path):
            print(f"Loading cached corpus embeddings+ids from {emb_path}")
            corpus_embeddings, doc_ids = load_dense_embeddings(emb_path)
        else:
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
            t0 = time.time()
            corpus_embeddings, doc_ids = stream_encode_corpus_dense(
                docs_iter,
                encoder,
                is_query=False,
                batch_size=2048,
                use_titles=True,
            )
            timing["doc_encoding_seconds"] += time.time() - t0

            if dataset_id == "irds:msmarco-passage/dev/small":
                save_dense_embeddings(corpus_embeddings, doc_ids, emb_path)

        # query embeddings: orig + HyDE
        t0 = time.time()
        q_orig = encoder.encode(texts=query_texts, is_query=True)
        q_hyde = encoder.encode(texts=hyde_queries, is_query=False)
        query_embeddings = 0.5 * (q_orig + q_hyde)
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

        del encoder, indexer, corpus_embeddings, query_embeddings, q_orig, q_hyde
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #############################
    # ---------------------------
    # 2–3. TART (Contriever retrieval + rerank)
    # ---------------------------
    #############################
    elif model_key == "tart":
        print(f"--- Running TART Reranker Pipeline with Contriever ---")

        retriever_encoder = DenseEncoder(MODELS["contriever"]["model_path"], MODELS["contriever"], device)

        out_dir = "outputs/embeddings/contriever"
        os.makedirs(out_dir, exist_ok=True)
        emb_path = os.path.join(out_dir, f"{corpus_label}.npz")

        # # Need doc_ids always
        # docs_iter_for_ids, _, _ = handler.read(corpus_id, variant=corpus_variant)
        # doc_ids = [str(d.get("doc_id", "")) for d in tqdm(docs_iter_for_ids, desc="Loading doc_ids")]

        # corpus embeddings
        if os.path.exists(emb_path):
            print(f"[TART] Loading cached corpus embeddings from {emb_path}")
            corpus_embeddings, doc_ids = load_dense_embeddings(emb_path)
        else:
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
            t0 = time.time()
            corpus_embeddings, _doc_ids_stream = stream_encode_corpus_dense(
                docs_iter,
                retriever_encoder,
                is_query=False,
                batch_size=2048,
                use_titles=True,
            )
            timing["doc_encoding_seconds"] += time.time() - t0
            if dataset_id == "irds:msmarco-passage/dev/small":
                save_dense_embeddings(corpus_embeddings, doc_ids, emb_path)

        indexer = FaissIndexer(dimension=corpus_embeddings.shape[1])
        t0 = time.time()
        indexer.build(corpus_embeddings)
        timing["index_build_seconds"] += time.time() - t0

        t0 = time.time()
        query_embeddings = retriever_encoder.encode(texts=query_texts, is_query=True)
        timing["query_encoding_seconds"] += time.time() - t0

        t0 = time.time()
        scores_1st, indices_1st = indexer.search(query_embeddings, top_k=top_k)
        timing["search_seconds"] += time.time() - t0

        # Build a small doc_map only for retrieved docs (stream texts once)
        print("[TART] Building small doc_map for reranking (streaming corpus once)...")
        needed_doc_ids = set()
        for row in indices_1st:
            for idx in row:
                needed_doc_ids.add(doc_ids[int(idx)])

        doc_map: Dict[str, str] = {}
        docs_iter_texts, _, _ = handler.read(corpus_id, variant=corpus_variant)
        for d in tqdm(docs_iter_texts, desc="Streaming corpus for doc_map"):
            did = str(d.get("doc_id", ""))
            if did in needed_doc_ids:
                doc_map[did] = d.get("text", "") or ""

        # free stage-1 heavy stuff
        del retriever_encoder, corpus_embeddings, indexer, query_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 2 rerank
        tart_encoder = DenseEncoder(model_name, config, device)
        instruction = config.get("instruction", "Find the passage that answers the given query.")

        results = {}
        rerank_start = time.time()
        for i, (qid, qtext) in tqdm(enumerate(zip(query_ids, query_texts)), total=len(query_ids), desc="TART Reranking"):
            cand_indices = indices_1st[i]
            cand_ids = [doc_ids[idx] for idx in cand_indices]
            cand_texts = [doc_map.get(did, "") for did in cand_ids]

            scores_list = tart_encoder.rerank(
                queries=[qtext],
                docs_per_query=[cand_texts],
                instruction=instruction,
                batch_size=16,
            )
            results[qid] = {did: float(s) for did, s in zip(cand_ids, scores_list[0])}
        timing["rerank_seconds"] += time.time() - rerank_start

        del tart_encoder, doc_map, doc_ids, indices_1st
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #############################
    # ---------------------------
    # 2–3. Generic dense retrieval
    # ---------------------------
    #############################
    else:
        encoder = DenseEncoder(model_name, config, device)

        out_dir = f"outputs/embeddings/{model_key}"
        os.makedirs(out_dir, exist_ok=True)
        emb_path = os.path.join(out_dir, f"{corpus_label}.npz")

        if os.path.exists(emb_path):
            print(f"Loading cached corpus embeddings+ids from {emb_path}")
            corpus_embeddings, doc_ids = load_dense_embeddings(emb_path)
        else:
            docs_iter, _, _ = handler.read(corpus_id, variant=corpus_variant)
            t0 = time.time()
            corpus_embeddings, doc_ids = stream_encode_corpus_dense(
                docs_iter,
                encoder,
                is_query=False,
                batch_size=2048,
                use_titles=True,
            )
            timing["doc_encoding_seconds"] += time.time() - t0

            if model_key == "contriever" or dataset_id == "irds:msmarco-passage/dev/small":
                save_dense_embeddings(corpus_embeddings, doc_ids, emb_path)

        # query embeddings
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

        del encoder, indexer, corpus_embeddings, query_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------
    # 4. Save & Evaluate
    # ---------------------------
    out_dir = f"outputs/results/{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{dataset_label}.json")
    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Saved search results to {results_path}")

    evaluator = Evaluator(dataset_id, skip_self_matches="auto")

    eval_start = time.time()
    metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels, results)
    evaluation_seconds = time.time() - eval_start

    print(f"\n--- Results for model {model_key} ---")
    print(pd.DataFrame([metrics_agg]))

    elapsed = time.time() - start_time

    # Free up RAM
    if "jhu-clsp" not in dataset_id.lower():
        del results
        gc.collect()
        results = None

    avg_latency_ms = (timing["search_seconds"] / timing["num_queries"]) * 1000.0 if timing["num_queries"] > 0 else None
    timing["avg_latency_ms"] = avg_latency_ms
    timing["evaluation_seconds"] = float(evaluation_seconds)
    timing["runtime_seconds"] = float(elapsed)

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

    del evaluator, queries, qrels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------
    # 6. Archival (unchanged logic, but keep it after big deletes)
    # ---------------------------
    if archive:
        ARCHIVE_ROOT = "/dataHDD1/masterthesis"
        artefacts = []

        sparse_keys = ["bm25", "doc2query", "unicoil", "sparta", "deepct", "splade"]

        if model_key in sparse_keys:
            enc_dir = f"outputs/encodings/{model_key}/{corpus_label}"
            index_dir = f"outputs/indexes/{model_key}/{corpus_label}"
            if os.path.exists(enc_dir):
                artefacts.append(enc_dir)
            if os.path.exists(index_dir):
                artefacts.append(index_dir)

        if model_key == "hyde" or (model_key not in sparse_keys and model_key != "colbert"):
            emb_path = f"outputs/embeddings/{model_key}/{corpus_label}.npz"
            if os.path.exists(emb_path):
                artefacts.append(emb_path)

        if model_key == "colbert":
            colbert_index_dir = f"colbert/indexes/{dataset_label}"
            if os.path.exists(colbert_index_dir):
                artefacts.append(colbert_index_dir)

        result_file = f"outputs/results/{model_key}/{dataset_label}.json"
        if os.path.exists(result_file):
            artefacts.append(result_file)

        for folder in ["embeddings", "encodings", "indexes", "results"]:
            path = os.path.join(ARCHIVE_ROOT, f"outputs/{folder}/{model_key}")
            os.makedirs(path, exist_ok=True)

        for artefact in artefacts:
            if os.path.isdir(artefact):
                dest_dir = os.path.join(ARCHIVE_ROOT, artefact)
                os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                print(f"[ARCHIVE] Copying folder → {artefact} → {dest_dir}")
                shutil.copytree(artefact, dest_dir, dirs_exist_ok=True)
            elif os.path.isfile(artefact):
                dest_file = os.path.join(ARCHIVE_ROOT, artefact)
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                print(f"[ARCHIVE] Copying file → {artefact} → {dest_file}")
                shutil.copy2(artefact, dest_file)

            # delete from NVMe (same condition you had)
            if corpus_id not in ["irds:msmarco-passage/dev", "irds:msmarco-passage/dev/small"]:
                if os.path.isdir(artefact):
                    print(f"[CLEANUP] Removing folder from NVMe → {artefact}")
                    shutil.rmtree(artefact)
                elif os.path.isfile(artefact):
                    print(f"[CLEANUP] Removing file from NVMe → {artefact}")
                    os.remove(artefact)

    return {
        "metrics_agg": metrics_agg,
        "metrics_perq": metrics_perq,
        "summary_stats": summary_stats,
        "runtime_seconds": float(elapsed),
        "variant": variant,
        "timing": timing,
    }
