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
from typing import List, Tuple, Dict, Any, Iterable, Optional
import pandas as pd
import pyarrow.dataset as ds
import ir_datasets
import datasets as hf_datasets


class DataHandler:
    """
    Handles the downloading, saving, and efficient loading of all datasets.
    
    sources: list like ["irds:beir/trec-covid", "hf:jhu-clsp/news21-instructions"]
    folder: root directory; each dataset (and variant) gets its own subfolder
    """
    
    def __init__(self, sources: List[str], folder: str, compression: str = "zstd"):
        self.sources = sources
        self.folder = folder
        self.compression = compression

    # -------
    # Helpers
    # -------
    def _pick_col(self, df: pd.DataFrame, candidates: List[str], new_name: str) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                return df[c].rename(new_name)
        return pd.Series([pd.NA] * len(df), name=new_name)

    def _normalize_docs(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "doc_id": self._pick_col(df, ["doc_id", "id", "_id", "document_id", "corpus-id"], "doc_id"),
            "text": self._pick_col(df, ["text", "contents", "document"], "text"),
            "title": self._pick_col(df, ["title"], "title"),
        })

    def _normalize_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "query_id": self._pick_col(df, ["query_id", "id", "_id", "query-id"], "query_id"),
            "text": self._pick_col(df, ["text", "query"], "text"),
        })

    def _normalize_qrels(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "qid": self._pick_col(df, ["query_id", "qid", "query-id"], "qid"),
            "docno": self._pick_col(df, ["doc_id", "docno", "_id", "corpus-id"], "docno"),
            "label": self._pick_col(df, ["relevance", "label", "score"], "label"),
        })

    def _to_pandas(self, obj) -> pd.DataFrame:
        if isinstance(obj, hf_datasets.DatasetDict):
            return pd.concat([split.to_pandas() for split in obj.values()], ignore_index=True)
        elif isinstance(obj, hf_datasets.Dataset):
            return obj.to_pandas()
        else:
            raise ValueError(f"Unexpected HF dataset type: {type(obj)}")

    # ----------------------------
    # Load + normalize one dataset
    # ----------------------------
    def _load_one(self, source: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Returns a dict mapping variant -> splits dict (each with keys: docs, queries, qrels).

        Non-jhu-clsp:
            {"base": {"docs", "queries", "qrels"}}

        jhu-clsp (generic):
            A subset of {"og": {...}, "changed": {...}} depending on which
            columns/splits actually exist in the dataset.
        """
        typ, name = source.split(":", 1)

        if typ == "irds":
            irds_ds = ir_datasets.load(name)
            docs = self._normalize_docs(pd.DataFrame(irds_ds.docs_iter()))
            queries = self._normalize_queries(pd.DataFrame(irds_ds.queries_iter()))
            qrels = self._normalize_qrels(pd.DataFrame(irds_ds.qrels_iter()))
            return {"base": {"docs": docs, "queries": queries, "qrels": qrels}}

        if typ != "hf":
            raise ValueError(f"Unknown source type: {typ}")

        # HF path
        corpus_df = self._to_pandas(hf_datasets.load_dataset(name, "corpus"))
        docs = self._normalize_docs(corpus_df)

        is_jhu = ("jhu-clsp" in name.lower())

        if is_jhu:
            out: Dict[str, Dict[str, pd.DataFrame]] = {}
            # queries split is single place where we can check for OG/CHANGED text columns
            queries_df = self._to_pandas(hf_datasets.load_dataset(name, "queries"))

            # OG
            if "_id" in queries_df.columns and "instruction_og" in queries_df.columns:
                qo = queries_df[["_id", "text", "instruction_og"]].copy()
                qo["query_id"] = qo["_id"].astype(str)

                # FollowIR default: query + space + instruction
                qo["text"] = (
                    qo["text"].fillna("").astype(str).str.strip()
                    + " "
                    + qo["instruction_og"].fillna("").astype(str).str.strip()
                ).str.strip()

                queries_og = self._normalize_queries(qo[["query_id", "text"]])

                try:
                    qrels_og_df = self._to_pandas(hf_datasets.load_dataset(name, "qrels_og"))
                    qrels_og = self._normalize_qrels(qrels_og_df)
                    out["og"] = {"docs": docs, "queries": queries_og, "qrels": qrels_og}
                except Exception:
                    pass

            # CHANGED
            if "_id" in queries_df.columns and "instruction_changed" in queries_df.columns:
                qc = queries_df[["_id", "text", "instruction_changed"]].copy()
                qc["query_id"] = qc["_id"].astype(str)

                qc["text"] = (
                    qc["text"].fillna("").astype(str).str.strip()
                    + " "
                    + qc["instruction_changed"].fillna("").astype(str).str.strip()
                ).str.strip()

                queries_changed = self._normalize_queries(qc[["query_id", "text"]])

                try:
                    qrels_changed_df = self._to_pandas(hf_datasets.load_dataset(name, "qrels_changed"))
                    qrels_changed = self._normalize_qrels(qrels_changed_df)
                    out["changed"] = {"docs": docs, "queries": queries_changed, "qrels": qrels_changed}
                except Exception:
                    pass

            # Fallback: if neither variant materialized, try standard queries/qrels as "base"
            if not out:
                try:
                    q_df = self._to_pandas(hf_datasets.load_dataset(name, "queries"))
                    r_df = self._to_pandas(hf_datasets.load_dataset(name, "qrels"))
                    out["base"] = {
                        "docs": docs,
                        "queries": self._normalize_queries(q_df),
                        "qrels": self._normalize_qrels(r_df),
                    }
                except Exception:
                    # last resort: empty base
                    out["base"] = {
                        "docs": docs,
                        "queries": pd.DataFrame(columns=["query_id", "text"]),
                        "qrels": pd.DataFrame(columns=["qid", "docno", "label"]),
                    }
            return out

        # Non-JHU default
        q_df = self._to_pandas(hf_datasets.load_dataset(name, "queries"))
        r_df = self._to_pandas(hf_datasets.load_dataset(name, "qrels"))
        return {
            "base": {
                "docs": docs,
                "queries": self._normalize_queries(q_df),
                "qrels": self._normalize_qrels(r_df),
            }
        }

    # ----------------------------------------------
    # Save each dataset separately (fast early-skip)
    # ----------------------------------------------
    def save(self):
        for src in self.sources:
            typ, name = src.split(":", 1)
            base = f"{typ}_{name.replace('/', '_')}"
            is_jhu = (typ == "hf" and "jhu-clsp" in name.lower())

            if is_jhu:
                # We assume that, if both variants exist, we can skip without loading.
                # If only one is missing (or dataset has only one variant), we'll load and write what's available.
                subdirs = {
                    "og": os.path.join(self.folder, f"{base}_og"),
                    "changed": os.path.join(self.folder, f"{base}_changed"),
                }
                required = {
                    v: [os.path.join(sd, f) for f in ("docs.parquet", "queries.parquet", "qrels.parquet")]
                    for v, sd in subdirs.items()
                }
                have_og = all(os.path.exists(p) for p in required["og"])
                have_changed = all(os.path.exists(p) for p in required["changed"])

                if have_og and have_changed:
                    print(f"✅ Skipping {src}, both variants already exist.")
                    continue

                # Create subdirs only for variants we'll try to write (we don't yet know which exist in HF)
                if not have_og:
                    os.makedirs(subdirs["og"], exist_ok=True)
                if not have_changed:
                    os.makedirs(subdirs["changed"], exist_ok=True)

                print(f"⬇️ Processing {src} (jhu-clsp) ...")
                splits_by_variant = self._load_one(src)  # may contain "og" and/or "changed", or "base" fallback

                # Write OG if present and needed
                if "og" in splits_by_variant and not have_og:
                    sd = subdirs["og"]
                    sp = splits_by_variant["og"]
                    sp["docs"].to_parquet(os.path.join(sd, "docs.parquet"), compression=self.compression, index=False)
                    sp["queries"].to_parquet(os.path.join(sd, "queries.parquet"), compression=self.compression, index=False)
                    sp["qrels"].to_parquet(os.path.join(sd, "qrels.parquet"), compression=self.compression, index=False)
                    with open(os.path.join(sd, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump({"source": src, "variant": "og"}, f, indent=2)
                    print(f"💾 Saved {src} (og) to {sd}")

                # Write CHANGED if present and needed
                if "changed" in splits_by_variant and not have_changed:
                    sd = subdirs["changed"]
                    sp = splits_by_variant["changed"]
                    sp["docs"].to_parquet(os.path.join(sd, "docs.parquet"), compression=self.compression, index=False)
                    sp["queries"].to_parquet(os.path.join(sd, "queries.parquet"), compression=self.compression, index=False)
                    sp["qrels"].to_parquet(os.path.join(sd, "qrels.parquet"), compression=self.compression, index=False)
                    with open(os.path.join(sd, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump({"source": src, "variant": "changed"}, f, indent=2)
                    print(f"💾 Saved {src} (changed) to {sd}")

                # If neither variant was present (rare), fall back to a single base folder
                if ("og" not in splits_by_variant) and ("changed" not in splits_by_variant):
                    subdir = os.path.join(self.folder, base)
                    os.makedirs(subdir, exist_ok=True)
                    sp = splits_by_variant["base"]
                    sp["docs"].to_parquet(os.path.join(subdir, "docs.parquet"), compression=self.compression, index=False)
                    sp["queries"].to_parquet(os.path.join(subdir, "queries.parquet"), compression=self.compression, index=False)
                    sp["qrels"].to_parquet(os.path.join(subdir, "qrels.parquet"), compression=self.compression, index=False)
                    with open(os.path.join(subdir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump({"source": src}, f, indent=2)
                    print(f"💾 Saved {src} (base fallback) to {subdir}")

            else:
                # Non-JHU, original fast path
                subdir = os.path.join(self.folder, base)
                docs_path = os.path.join(subdir, "docs.parquet")
                queries_path = os.path.join(subdir, "queries.parquet")
                qrels_path = os.path.join(subdir, "qrels.parquet")

                if all(os.path.exists(p) for p in (docs_path, queries_path, qrels_path)):
                    print(f"✅ Skipping {src}, files already exist in {subdir}")
                    continue

                os.makedirs(subdir, exist_ok=True)
                print(f"⬇️ Processing {src} ...")
                splits = self._load_one(src)["base"]
                splits["docs"].to_parquet(docs_path, compression=self.compression, index=False)
                splits["queries"].to_parquet(queries_path, compression=self.compression, index=False)
                splits["qrels"].to_parquet(qrels_path, compression=self.compression, index=False)
                with open(os.path.join(subdir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"source": src}, f, indent=2)
                print(f"💾 Saved {src} to {subdir}")

    # ---------------
    # Read in batches
    # ---------------
    def read(
        self,
        dataset: str,
        variant: Optional[str] = None,
        yield_batches: bool = False
    ) -> Tuple[Iterable[Dict[str, Any]], Iterable[Dict[str, Any]], Iterable[Dict[str, Any]]]:
        typ, name = dataset.split(":", 1)
        base = f"{typ}_{name.replace('/', '_')}"
        folder_name = base if not variant else f"{base}_{variant}"
        subdir = os.path.join(self.folder, folder_name)

        def scan(path):
            if yield_batches:
                for batch in ds.dataset(path, format="parquet").to_batches():
                    yield batch.to_pandas().to_dict(orient="records")
            else:
                for batch in ds.dataset(path, format="parquet").to_batches():
                    yield from batch.to_pandas().to_dict(orient="records")

        return (
            scan(os.path.join(subdir, "docs.parquet")),
            scan(os.path.join(subdir, "queries.parquet")),
            scan(os.path.join(subdir, "qrels.parquet")),
        )
