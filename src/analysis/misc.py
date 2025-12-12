import pandas as pd
import numpy as np
from typing import Optional
from src.analysis.mappings import DATASET_SIZES, MODEL_FAMILIES, MODEL_FAMILIES_PRETTY, MODEL_NAMES_PRETTY


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


def label_to_dataset_id_for_sizes(label: str) -> Optional[str]:
    ds = label_to_dataset_id(label)
    if ds is None:
        return None

    if ds in DATASET_SIZES:
        return ds

    if ds.endswith("/all"):
        parent = ds.rsplit("/", 1)[0]
        if parent in DATASET_SIZES:
            return parent

    return ds