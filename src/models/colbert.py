################################################################################
# Title
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import os
import json
import gzip
import time
import subprocess
import random
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src.evaluator import Evaluator
from src.configs.datasets import DATASETS
from src.misc import get_dataset_variants
from src.data_handler import DataHandler
from src.analysis.mappings import DATASET_SIZES

if not os.path.exists("ColBERT"):
    subprocess.check_call(["git", "clone", "https://github.com/stanford-futuredata/ColBERT.git"])

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

import faiss, torch
faiss.omp_set_num_threads(8)
torch.set_num_threads(8)


DATASET_MAPPING = {
    "irds:msmarco-passage/dev/small": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged": "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2020/judged": "irds:msmarco-passage/dev/small",
}


def parquet_to_tsv(docs_parquet: str, queries_parquet: str, qrels_parquet: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    docid_col = "doc_id"
    doctext_col = "text"
    doctitle_col = "title"

    qid_col = "query_id"
    qtext_col = "text"

    qrels_qid_col = "qid"
    qrels_docid_col = "docno"
    rel_col = "label"

    def clean_text(x) -> str:
        if x is None:
            return ""
        # Use split() and join() to collapse ALL whitespace (newlines, tabs, multiple spaces)
        return " ".join(str(x).split()).strip()

    # final paths
    collection_tsv = os.path.join(out_dir, "collection.tsv")
    pid2docid_tsv  = os.path.join(out_dir, "pid2docid.tsv")
    docid2pid_tsv  = os.path.join(out_dir, "docid2pid.tsv")
    queries_tsv    = os.path.join(out_dir, "queries.tsv")
    qrels_trec     = os.path.join(out_dir, "qrels.trec")

    # temp paths (atomic write)
    collection_tmp = collection_tsv + ".tmp"
    pid2docid_tmp  = pid2docid_tsv + ".tmp"
    docid2pid_tmp  = docid2pid_tsv + ".tmp"
    queries_tmp    = queries_tsv + ".tmp"
    qrels_tmp      = qrels_trec + ".tmp"

    # ---- docs -> collection.tsv + mappings ----
    pf_docs = pq.ParquetFile(docs_parquet)
    doc_cols = [docid_col, doctext_col] + ([doctitle_col] if doctitle_col else [])

    docid2pid = {}
    pid = 0

    with open(collection_tmp, "w", encoding="utf-8") as out_col, \
         open(pid2docid_tmp,  "w", encoding="utf-8") as out_p2d, \
         open(docid2pid_tmp,  "w", encoding="utf-8") as out_d2p:

        for batch in pf_docs.iter_batches(batch_size=100_000, columns=doc_cols):
            doc_ids = batch[docid_col].to_pylist()
            texts   = batch[doctext_col].to_pylist()
            titles  = batch[doctitle_col].to_pylist() if doctitle_col else [None] * len(doc_ids)

            for docid, title, text in zip(doc_ids, titles, texts):
                docid = str(docid)
                t = clean_text(text)
                if doctitle_col:
                    ttl = clean_text(title)
                    if ttl:
                        t = (ttl + " " + t).strip()

                # Always write TWO columns: pid<TAB>text
                out_col.write(f"{pid}\t{t}\n")
                out_p2d.write(f"{pid}\t{docid}\n")
                out_d2p.write(f"{docid}\t{pid}\n")

                docid2pid[docid] = pid
                pid += 1

    n_docs = pid

    # Validate the tmp collection BEFORE making it "official"
    with open(collection_tmp, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if "\t" not in line.rstrip("\n"):
                raise ValueError(f"Malformed collection.tsv line {i}: {repr(line[:200])}")

    # Atomic commit for docs outputs
    os.replace(collection_tmp, collection_tsv)
    os.replace(pid2docid_tmp, pid2docid_tsv)
    os.replace(docid2pid_tmp, docid2pid_tsv)

    # ---- queries -> queries.tsv ----
    pf_q = pq.ParquetFile(queries_parquet)
    n_queries = 0

    with open(queries_tmp, "w", encoding="utf-8") as out:
        for batch in pf_q.iter_batches(batch_size=100_000, columns=[qid_col, qtext_col]):
            qids = batch[qid_col].to_pylist()
            qtexts = batch[qtext_col].to_pylist()
            for qid, qt in zip(qids, qtexts):
                out.write(f"{str(qid)}\t{clean_text(qt)}\n")
                n_queries += 1

    os.replace(queries_tmp, queries_tsv)

    # ---- qrels -> qrels.trec (DOCNO -> PID) ----
    pf_r = pq.ParquetFile(qrels_parquet)
    n_qrels = 0
    missing = 0

    with open(qrels_tmp, "w", encoding="utf-8") as out:
        for batch in pf_r.iter_batches(batch_size=200_000, columns=[qrels_qid_col, qrels_docid_col, rel_col]):
            qids = batch[qrels_qid_col].to_pylist()
            docnos = batch[qrels_docid_col].to_pylist()
            rels = batch[rel_col].to_pylist()
            for qid, docno, rel in zip(qids, docnos, rels):
                docno = str(docno)
                pid = docid2pid.get(docno)
                if pid is None:
                    missing += 1
                    continue
                out.write(f"{str(qid)} 0 {pid} {int(rel)}\n")
                n_qrels += 1

    os.replace(qrels_tmp, qrels_trec)

    print(f"#Queries = {n_queries}")
    print(f"#Docs    = {n_docs}")
    print(f"#Qrels   = {n_qrels} (skipped {missing} qrels with missing docno)")
    print(f"Wrote: {collection_tsv}")
    print(f"Wrote: {pid2docid_tsv}")
    print(f"Wrote: {docid2pid_tsv}")
    print(f"Wrote: {queries_tsv}")
    print(f"Wrote: {qrels_trec}")


def _load_pid2docid(pid2docid_path: str) -> dict[str, str]:
    m = {}
    with open(pid2docid_path, "r", encoding="utf-8") as f:
        for line in f:
            pid, docid = line.rstrip("\n").split("\t", 1)
            m[pid] = docid
    return m


def _load_queries_tsv(queries_tsv: str) -> list[tuple[str, str]]:
    pairs = []
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for line in f:
            qid, qtext = line.rstrip("\n").split("\t", 1)
            pairs.append((qid, qtext))
    return pairs


def _load_qrels_parquet(qrels_parquet: str, qid_col: str = "qid", docno_col: str = "docno", rel_col: str = "label") -> pd.DataFrame:
    pf = pq.ParquetFile(qrels_parquet)
    rows = []
    for batch in pf.iter_batches(batch_size=200_000, columns=[qid_col, docno_col, rel_col]):
        qids = batch[qid_col].to_pylist()
        docnos = batch[docno_col].to_pylist()
        rels = batch[rel_col].to_pylist()
        rows.extend(zip(qids, docnos, rels))
    return pd.DataFrame(rows, columns=[qid_col, docno_col, rel_col])


def _get_args(dataset_label: str, dataset_sizes=DATASET_SIZES, nbits: int = 2):
    """
    Returns (nbits, ncells, ndocs) matching ColBERTv2 Appendix F-style settings,
    with a simple corpus-size heuristic that works across MS MARCO + BEIR + LoTTE.

    Mapping:
      - ncells ~= paper 'nprobe' (probe)
      - ndocs  ~= paper 'ncandidate' (candidates kept for exact scoring)
      - nbits  ~= residual bits per dim (b)
    """
    if dataset_label not in dataset_sizes:
        if dataset_label.startswith("irds:beir/cqadupstack/") or dataset_label.startswith("hf:jhu-clsp/"):
            nbits_out = 2
            ncells = 2
            ndocs = ncells * (2**12)
            return nbits_out, ncells, ndocs
        else:
            raise KeyError(f"Unknown dataset_label={dataset_label!r}. "
                        f"Add it to DATASET_SIZES or pass a different mapping.")

    num_docs = int(dataset_sizes[dataset_label]["docs"])

    # Paper-faithful: b=2 bits/dim is the common evaluation choice.
    # (You can override via the nbits parameter.)
    nbits_out = int(nbits)

    # Heuristic for probe (ncells):
    # - Default probe=2
    # - Use probe=4 for "large" corpora (Wikipedia/MSMARCO scale)
    # We approximate that with a doc-count threshold.
    if num_docs >= 5_000_000:
        ncells = 4
    else:
        ncells = 2

    # Heuristic for candidates (ndocs):
    # Paper says: ncandidate = nprobe * 2^12 by default,
    # with exceptions:
    #   - Wikipedia uses nprobe * 2^13
    #   - MS MARCO uses nprobe * 2^14
    #
    # We don't have "Wikipedia" explicitly in your mapping, so we do:
    #   - If it's MS MARCO passage: use 2^14
    #   - Else if it's huge (>=5M docs): use 2^13
    #   - Else: use 2^12
    if dataset_label.startswith("irds:msmarco-passage/"):
        ndocs = ncells * (2**14)
    elif num_docs >= 5_000_000:
        ndocs = ncells * (2**13)
    else:
        ndocs = ncells * (2**12)

    return nbits_out, ncells, ndocs


def run_colbert(
    dataset_id: str,
    dataset_label: str,
    dataset_dir: str,
    qrels_parquet: str,
    index_root: str,
    index_name: str,
    checkpoint: str = "colbert-ir/colbertv2.0",
    gpus: int = 1,
    top_k: int = 1001,
    overwrite: str | bool = "reuse",   # "reuse" / "resume" / True
    save_report: bool = True,
) -> dict:
    """
    Mirrors run.py behavior:
      - skips if outputs/scores/colbert/<dataset_label>.json exists
      - saves results to outputs/results/colbert/<dataset_label>.json.gz
      - evaluates and saves report to outputs/scores/colbert/<dataset_label>.json

    Returns: dict similar to run.py return payload.
    """
    score_path = f"outputs/scores/colbert/{dataset_label}.json"
    if os.path.exists(score_path):
        print(f"Score found at {score_path}. Skipping...")
        return {}

    # ---------- timing structure like run.py ----------
    timing = {
        "doc_encoding_seconds": 0.0,    # ColBERT doc encoding is inside index
        "index_build_seconds": 0.0,     # ColBERT index build
        "query_encoding_seconds": 0.0,  # ColBERT query encoding is inside search
        "search_seconds": 0.0,          # search
        "rerank_seconds": 0.0,
        "num_queries": 0,
    }

    os.makedirs("outputs/results/colbert", exist_ok=True)
    os.makedirs("outputs/scores/colbert", exist_ok=True)

    collection_tsv = os.path.join(dataset_dir, "collection.tsv")
    queries_tsv = os.path.join(dataset_dir, "queries.tsv")
    pid2docid_tsv = os.path.join(dataset_dir, "pid2docid.tsv")

    assert os.path.exists(collection_tsv), f"Missing {collection_tsv}"
    assert os.path.exists(queries_tsv), f"Missing {queries_tsv}"
    assert os.path.exists(pid2docid_tsv), f"Missing {pid2docid_tsv}"

    print(f"--- Running Benchmark ---")
    print(f"Model: colbert | Checkpoint: {checkpoint} | Dataset: {dataset_label}")

    start_time = time.time()

    # ---------- load queries + mapping ----------
    pid2docid = _load_pid2docid(pid2docid_tsv)
    queries = _load_queries_tsv(queries_tsv)
    timing["num_queries"] = len(queries)
    
    # ---------- make ColBERT config -------------
    query_maxlen = 32
    if "arguana" in dataset_label:
        query_maxlen = 300
    elif "climate-fever" in dataset_label:
        query_maxlen = 64
        
    nbits, ncells, ndocs = _get_args(dataset_id)
    
    config = ColBERTConfig(
        checkpoint=checkpoint,
        index_root=index_root,
        nbits=nbits,
        query_maxlen=query_maxlen,
        doc_maxlen=300,
        kmeans_niters=4,
        index_bsize=128,
        ncells=ncells,
        ndocs=ndocs,
    )

    # ---------- 1) index ----------
    t0 = time.time()
    with Run().context(RunConfig(nranks=gpus, gpus=gpus)):
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection_tsv, overwrite=overwrite)
    timing["index_build_seconds"] += time.time() - t0

    # ---------- 2) search ----------
    t0 = time.time()
    with Run().context(RunConfig(nranks=1, gpus=1)):
        searcher = Searcher(index=index_name, config=config)

        results = {}
        for qid, qtext in tqdm(queries, desc="ColBERT search", total=len(queries)):
            pids, ranks, scores = searcher.search(qtext, k=top_k)

            q_res = {}
            for pid, score in zip(pids, scores):
                pid_str = str(pid)
                docid = pid2docid.get(pid_str)
                if docid is None:
                    continue
                q_res[docid] = float(score)
            results[str(qid)] = q_res

    timing["search_seconds"] += time.time() - t0

    # ---------- 3) save results (gz json) ----------
    results_path = f"outputs/results/colbert/{dataset_label}.json.gz"
    with gzip.open(results_path, "wt", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Saved search results to {results_path}")

    # ---------- 4) evaluate like run.py ----------
    qrels_df = _load_qrels_parquet(qrels_parquet, qid_col="qid", docno_col="docno", rel_col="label")
    # Ensure strings like your run.py
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["docno"] = qrels_df["docno"].astype(str)

    evaluator = Evaluator(dataset_id, skip_self_matches="auto")
    eval_start = time.time()
    metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels_df, results)
    evaluation_seconds = time.time() - eval_start

    elapsed = time.time() - start_time

    avg_latency_ms = (timing["search_seconds"] / timing["num_queries"]) * 1000.0 if timing["num_queries"] else None
    timing["avg_latency_ms"] = avg_latency_ms
    timing["evaluation_seconds"] = float(evaluation_seconds)
    timing["runtime_seconds"] = float(elapsed)

    report = {
        "model_name": "colbert",
        "dataset_id": dataset_id,
        "variant": None,
        "metrics": metrics_agg,
        "summary_stats": summary_stats,
        "runtime_seconds": float(elapsed),
        "timing": timing,
    }

    if save_report:
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(report, f)
        print(f"Saved metrics report to {score_path}")

    return {
        "metrics_agg": metrics_agg,
        "metrics_perq": metrics_perq,
        "summary_stats": summary_stats,
        "results": results,
        "runtime_seconds": float(elapsed),
        "variant": None,
        "timing": timing,
    }


def acquire_lock(lock_path: str, wait_s: int = 0):
    # atomic create; fails if exists
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if wait_s and (time.time() - start) < wait_s:
                time.sleep(1)
                continue
            raise RuntimeError(f"Lock exists: {lock_path} (another process is writing TSVs)")


##########################################################################
# Main loop
##########################################################################
if __name__ == "__main__":

    handler = DataHandler(
        sources=DATASETS,
        folder="data/raw"
    )

    model = "colbert"

    results_per_query = {}
    runs_cache = {}

    for dataset in DATASETS:
        base_label = dataset.replace("/", "_").replace(":", "_")
        run_key = (dataset, model)

        for variant in get_dataset_variants(handler, dataset):
            print(f"\n▶ Running {model} on {dataset}; variant: {variant}", flush=True)

            # Define paths including jhu-clsp variants handling
            suffix = f"_{variant}" if variant is not None else ""
            raw_dir = f"data/raw/{base_label}{suffix}"
            dataset_dir = f"data/colbert/{base_label}{suffix}"

            dataset_label = f"{base_label}{suffix}"
            index_root = "outputs/indexes/colbert"
            index_name = dataset_label
            qrels_parquet = f"{raw_dir}/qrels.parquet"

            # Turn parquet files into TSV files
            if not os.path.exists(dataset_dir):
                lock_path = os.path.join(dataset_dir, ".build_tsv.lock")
                os.makedirs(dataset_dir, exist_ok=True)

                acquire_lock(lock_path)
                try:
                    parquet_to_tsv(
                        docs_parquet=f"data/raw/{base_label}/docs.parquet",
                        queries_parquet=f"data/raw/{base_label}/queries.parquet",
                        qrels_parquet=f"data/raw/{base_label}/qrels.parquet",
                        out_dir=dataset_dir,
                    )
                finally:
                    os.remove(lock_path)

            # Run ColBERT
            index_root = "outputs/indexes/colbert"
            index_name = dataset_label

            out = run_colbert(
                dataset_id=dataset,
                dataset_label=dataset_label,
                dataset_dir=dataset_dir,
                qrels_parquet=qrels_parquet,
                index_root=index_root,
                index_name=index_name,
                gpus=1,
                top_k=1001,
                overwrite=True,
                save_report=True,
            )

            bucket = runs_cache.setdefault(run_key, {})
            bucket[variant] = out

            if set(bucket.keys()) >= {"og", "changed"}:
                evaluator = Evaluator(dataset, skip_self_matches="auto")

                qrels_og = handler.read(dataset, variant="og")[2]
                qrels_ch = handler.read(dataset, variant="changed")[2]
                qrels_og_df = pd.DataFrame(list(qrels_og))
                qrels_ch_df = pd.DataFrame(list(qrels_ch))

                run_og = bucket["og"]["results"]
                run_ch = bucket["changed"]["results"]

                p_mrr_macro, p_mrr_perq = evaluator.p_mrr(qrels_og_df, qrels_ch_df, run_og, run_ch, k=None)
                print(f"[{model} | {dataset}] p-MRR = {p_mrr_macro*100:.3f}", flush=True)

                og_agg = bucket["og"]["metrics_agg"]
                if "ndcg_cut_5" in og_agg:
                    std_name = "ndcg_cut_5"
                elif "mean_avg_precision" in og_agg:
                    std_name = "mean_avg_precision"
                else:
                    std_name = sorted(og_agg.keys())[0] if og_agg else "standard_metric"

                std_value = float(og_agg.get(std_name, float("nan")))

                elapsed_og = bucket["og"]["timing"]
                elapsed_ch = bucket["changed"]["timing"]

                out_dir = f"outputs/scores/{model}"
                os.makedirs(out_dir, exist_ok=True)
                combined_path = os.path.join(out_dir, f"{base_label}.json")

                combined_report = {
                    "model_name": model,
                    "dataset_id": dataset,
                    "metrics": {std_name: std_value, "p_mrr": float(p_mrr_macro)},
                    "summary_stats": {
                        "og": bucket["og"]["summary_stats"],
                        "changed": bucket["changed"]["summary_stats"],
                    },
                    "runtime": {
                        "og": elapsed_og,
                        "changed": elapsed_ch,
                    }
                }

                with open(combined_path, "w", encoding="utf-8") as f:
                    json.dump(combined_report, f, indent=2)
                print(f"💾 Saved combined report to {combined_path}", flush=True)
