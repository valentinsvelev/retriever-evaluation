import os
import pandas as pd
import numpy as np
from collections import Counter
import hashlib
import re
import torch
import ir_datasets as irds
from src.data_handler import DataHandler


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


def save_dense_embeddings(embeddings: torch.Tensor, doc_ids: list[str], path: str) -> None:
    arr = embeddings.detach().cpu().numpy().astype("float16")
    np.savez_compressed(path, embeddings=arr, doc_ids=np.array(doc_ids, dtype=object))


def load_dense_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    emb = torch.from_numpy(data["embeddings"].astype("float32"))
    if "doc_ids" in data.files:
        return emb, data["doc_ids"].tolist()
    # old cache format
    return emb, None
