################################################################################
# misc.py
#
# Description: Provides shared utility functions for the analysis pipeline, 
# including bootstrap statistics, latency computation, model/dataset name 
# normalization, token-length estimation, average document/query length 
# calculation, and benchmark label mapping.
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import pandas as pd
import numpy as np
import string
from typing import Optional
from src.data_handler import DataHandler
from src.analysis.mappings import MODEL_FAMILIES, MODEL_FAMILIES_PRETTY, MODEL_NAMES_PRETTY


def bootstrap_mean_stats(values, n_boot=100000, ci=0.95, random_state=42):
    s = pd.Series(values).dropna().to_numpy()
    n = s.shape[0]
    if n == 0:
        return {
            "standard_dev": float("nan"),
            "standard_error": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    if n == 1:
        v = float(s[0])
        return {
            "standard_dev": 0.0,
            "standard_error": 0.0,
            "ci95_low": v,
            "ci95_high": v,
        }

    # per-query variability
    sd = float(s.std(ddof=1))

    # bootstrap mean distribution
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = s[idx].mean(axis=1)

    se = float(boot_means.std(ddof=1))
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))

    return {
        "standard_dev": sd,
        "standard_error": se,
        "ci95_low": low,
        "ci95_high": high,
    }


def compute_latency_ms(rt):
    if rt is None or not isinstance(rt, dict):
        return np.nan

    if "avg_latency_ms" in rt and rt["avg_latency_ms"] is not None:
        return rt["avg_latency_ms"]

    # if all timing components are missing, return NaN
    components = [
        rt.get("query_encoding_seconds"),
        rt.get("search_seconds"),
        rt.get("rerank_seconds"),
    ]
    if all(v is None for v in components):
        return np.nan

    T_query = sum(v for v in components if v is not None)
    num_q = rt.get("num_queries", 1)
    if not num_q:
        return np.nan

    return T_query / num_q * 1000.0


def assign_family(model_name: str):
    for family, keywords in MODEL_FAMILIES_PRETTY.items():
        for kw in keywords:
            if kw in model_name:
                return family
    return "other"


def prettify_model_name(model_name: str):
    for key, pretty in MODEL_NAMES_PRETTY.items():
        if key in model_name:
            return pretty
    return model_name


def label_to_dataset_id(label: str) -> Optional[str]:
    """
    Inverse of: dataset_label = dataset_id.replace('/', '_').replace(':', '_')
    Examples:
      'irds_beir_cqadupstack_android' -> 'irds:beir/cqadupstack/android'
      'irds_msmarco-passage_dev'      -> 'irds:msmarco-passage/dev'
      'hf_jhu-clsp_instructir'        -> 'hf:jhu-clsp/instructir'
    """
    if label is None:
        return None

    s = str(label)

    # already a canonical id
    if ":" in s:
        return s

    # nothing to split on -> just return as-is
    if "_" not in s:
        return s

    # first '_' becomes ':', the rest become '/'
    prefix, rest = s.split("_", 1)
    dataset_id = f"{prefix}:{rest.replace('_', '/')}"

    return dataset_id


PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def count_words_no_punct(text: str) -> int:
    """
    Removes punctuation, splits on whitespace, counts tokens.
    """
    if not isinstance(text, str):
        return 0
    cleaned = text.translate(PUNCT_TABLE)
    # You could lower() if you want, but it's not required for counting
    return len(cleaned.split())


def compute_avg_lengths(data_handler: DataHandler, dataset: str, variant=None):
    doc_iter, query_iter, _ = data_handler.read(dataset, variant)

    # --- Docs ---
    doc_total, doc_count = 0, 0
    for row in doc_iter:
        txt = row.get("text")
        n = count_words_no_punct(txt)
        doc_total += n
        doc_count += 1
    avg_doc = doc_total / doc_count if doc_count else np.nan

    # --- Queries ---
    query_total, query_count = 0, 0
    for row in query_iter:
        txt = row.get("text")
        n = count_words_no_punct(txt)
        query_total += n
        query_count += 1
    avg_query = query_total / query_count if query_count else np.nan

    return avg_doc, avg_query


def compute_all_avg_lengths(handler: DataHandler, dataset_ids):
    results = {}

    for ds in dataset_ids:
        is_jhu = ("jhu-clsp" in ds.lower())
        results[ds] = {}

        if is_jhu:
            # variants: og and changed
            for variant in ["og", "changed"]:
                try:
                    avg_doc, avg_query = compute_avg_lengths(handler, ds, variant=variant)
                    results[ds][variant] = {
                        "avg_doc_len": avg_doc,
                        "avg_query_len": avg_query,
                    }
                except Exception as e:
                    # dataset might not have both variants
                    print(f"⚠️ Skipping {ds} variant={variant}: {e}")
        else:
            # normal dataset: single base split
            try:
                avg_doc, avg_query = compute_avg_lengths(handler, ds, variant=None)
                results[ds]["base"] = {
                    "avg_doc_len": avg_doc,
                    "avg_query_len": avg_query,
                }
            except Exception as e:
                print(f"⚠️ Skipping {ds}: {e}")

    return results


def get_benchmark(dataset_id):
    dataset_id = dataset_id.lower()
    if "jhu-clsp" in dataset_id:
        return "FollowIR"
    elif "kaist-ai" in dataset_id:
        return "InstructIR"
    elif "beir" in dataset_id:
        return "BEIR"
    elif "lotte" in dataset_id:
        return "LoTTE"
    elif "trec-dl-2019" in dataset_id:
        return "TREC-DL 2019"
    elif "trec-dl-2020" in dataset_id:
        return "TREC-DL 2020"
    elif "msmarco-passage" in dataset_id:
        return "MS MARCO"
    else:
        print("UNKNOWN")
        return "UNKNOWN"