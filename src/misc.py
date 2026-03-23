################################################################################
# misc.py
#
# Description: Provides utility functions for directory setup, corpus preparation, 
# dataset variant detection, and saving/loading dense embedding caches in NumPy, 
# JSONL, and HDF5 formats.
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import hashlib
import re
import torch
import ir_datasets as irds
from src.data_handler import DataHandler
import h5py


def create_folder_structure(only_local: bool = False):
    BASE_DIR = "/dataHDD1"

    hdd_paths = [
        "masterthesis",
        "masterthesis/data",
        "masterthesis/data/raw",
        "masterthesis/data/corpora",
        "masterthesis/data/augmented",
        "masterthesis/outputs",
        "masterthesis/outputs/results",
        "masterthesis/outputs/scores",
        "masterthesis/outputs/indexes",
        "masterthesis/outputs/embeddings",
        "masterthesis/outputs/encodings"
    ]

    nvme_paths = [
        "data",
        "data/raw",
        "data/corpora",
        "data/augmented",
        "outputs",
        "outputs/results",
        "outputs/scores",
        "outputs/indexes",
        "outputs/embeddings",
        "outputs/encodings"
    ]

    if only_local:
        paths = nvme_paths
    else:
        paths = [f"{BASE_DIR}/{p}" for p in hdd_paths] + nvme_paths

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Folder {path} created.")
        else:
            print(f"Folder {path} already exists. Skipping ...")


def prepare_pyserini_corpus(df: pd.DataFrame, corpus_dir: str, dataset_label: str):
    if "title" in df.columns and not df["title"].isnull().all():
        print("Concatenating titles and texts...")
        df["text"] = df["title"] + " " + df["text"]
    pyserini_df = df.rename(columns={'doc_id': 'id', 'text': 'contents'})
    pyserini_df["text"] = pyserini_df["contents"]
    os.makedirs(corpus_dir, exist_ok=True)
    output_path = os.path.join(corpus_dir, "corpus.jsonl")
    pyserini_df.to_json(output_path, orient='records', lines=True) # compression="zstd"
    print(f"Created Pyserini-compatible corpus at {output_path}")


def get_dataset_variants(handler: DataHandler, dataset_id: str):
    typ, name = dataset_id.split(":", 1)
    base = f"{typ}_{name.replace('/', '_')}"
    root = handler.folder

    def have(folder):
        return all(os.path.exists(os.path.join(folder, f)) for f in ("docs.parquet", "queries.parquet", "qrels.parquet"))

    og_dir = os.path.join(root, f"{base}_og")
    ch_dir = os.path.join(root, f"{base}_changed")

    has_og = have(og_dir)
    has_ch = have(ch_dir)

    if has_og or has_ch:
        out = []
        if has_og: out.append("og")
        if has_ch: out.append("changed")
        return out

    return [None] # normal dataset (all but FollowIR)


def save_dense_embeddings(embeddings, doc_ids: list[str], path: str) -> None:
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        arr = embeddings
    else:
        raise TypeError(f"Unsupported type: {type(embeddings)}")

    arr = arr.astype("float16")
    np.savez_compressed(
        path,
        embeddings=arr,
        doc_ids=np.array(doc_ids, dtype=object)
    )


def load_dense_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    emb = torch.from_numpy(data["embeddings"].astype("float32"))
    if "doc_ids" in data.files:
        return emb, data["doc_ids"].tolist()
    # old cache format
    return emb, None


def load_dense_embeddings_hdf5(path, as_tensor=False):
    """
    Load dense embeddings and document IDs from an HDF5 file.
    """
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:] # (N, D), float16
        doc_ids = f["doc_ids"][:]

    # Decode byte strings -> Python strings
    doc_ids = [doc_id.decode("utf-8") for doc_id in doc_ids]

    if as_tensor:
        embeddings = torch.from_numpy(embeddings)

    return embeddings, doc_ids


def iter_dense_embeddings_hdf5(path, batch_size=8192, decode_doc_ids=True):
    with h5py.File(path, "r") as f:
        emb_ds = f["embeddings"]
        id_ds  = f["doc_ids"]
        n = emb_ds.shape[0]

        for i in range(0, n, batch_size):
            emb = emb_ds[i:i+batch_size]  # only this slice is loaded
            ids = id_ds[i:i+batch_size]
            if decode_doc_ids:
                ids = [x.decode("utf-8") for x in ids]
            yield emb, ids


def get_hdf5_embedding_dim(path):
    with h5py.File(path, "r") as f:
        return f["embeddings"].shape[1]


def append_dense_embeddings_jsonl(embeddings, doc_ids, path):
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        arr = embeddings
    else:
        raise TypeError(type(embeddings))

    arr = arr.astype("float16")

    with open(path, "a", encoding="utf-8") as f:
        for emb, doc_id in zip(arr, doc_ids):
            rec = {
                "doc_id": doc_id,
                "embedding": emb.tolist()
            }
            f.write(json.dumps(rec) + "\n")


def append_dense_embeddings_hdf5(embeddings, doc_ids, path):
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
    else:
        arr = np.asarray(embeddings)

    arr = arr.astype("float16")

    with h5py.File(path, "a") as f:
        if "embeddings" not in f:
            f.create_dataset(
                "embeddings",
                data=arr,
                maxshape=(None, arr.shape[1]),
                chunks=True,
                compression="gzip",
            )
            f.create_dataset(
                "doc_ids",
                data=np.array(doc_ids, dtype="S"),
                maxshape=(None,),
                chunks=True,
                compression="gzip",
            )
        else:
            n = f["embeddings"].shape[0]
            f["embeddings"].resize(n + arr.shape[0], axis=0)
            f["embeddings"][n:] = arr

            f["doc_ids"].resize(n + len(doc_ids), axis=0)
            f["doc_ids"][n:] = np.array(doc_ids, dtype="S")