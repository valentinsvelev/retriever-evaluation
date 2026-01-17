import os
import sys
import json
import re
import math
import traceback
from typing import Dict, Any, Iterable, Tuple, List, Optional

import torch
import torch.multiprocessing as mp
import pandas as pd
from tqdm import tqdm

from pyserini.encode import (
    SpladeQueryEncoder,
    UniCoilQueryEncoder,
    UniCoilDocumentEncoder,
)
from sentence_transformers import SparseEncoder as STSparseEncoder
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForTokenClassification
)

from src.models.sparta import SPARTA


# -------------------------
# Multiprocess helpers
# -------------------------

def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def _iter_corpus_strided(
    in_path: str,
    rank: int,
    world: int,
    *,
    expected: int | None = None,
    desc: str = "",
    position: int = 0,
) -> Iterable[Tuple[str, str]]:
    """Yield (doc_id, text) for lines where line_index % world == rank."""
    pbar = tqdm(total=expected, desc=desc, position=position, leave=False) if expected else None
    try:
        with open(in_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                if i % world != rank:
                    continue
                obj = json.loads(line)
                docid = str(obj.get("id") or obj.get("doc_id") or obj.get("_id") or "")
                text = (obj.get("text") or obj.get("contents") or "").strip()
                if not (docid and text):
                    continue
                if pbar:
                    pbar.update(1)
                yield docid, text
    finally:
        if pbar:
            pbar.close()


def _merge_parts(out_path: str, world: int) -> None:
    """Merge out_path.part{r} into out_path, then delete parts."""
    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in range(world):
            part = out_path + f".part{r}"
            with open(part, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
    for r in range(world):
        try:
            os.remove(out_path + f".part{r}")
        except OSError:
            pass


def _spawn_multi_gpu(
    *,
    mode: str,
    in_path: str,
    out_path: str,
    world: int,
    worker_kwargs: Dict[str, Any],
) -> None:
    """
    Spawn one process per GPU, each writing a part file, then merge.
    Uses mp.spawn for better error propagation.
    """
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(out_path))[0]

    for r in range(world):
        p = os.path.join(out_dir, f"{base}_shard{r:02d}.jsonl")
        if os.path.exists(p):
            os.remove(p)

    job = {
        "mode": mode,
        "in_path": in_path,
        "out_path": out_path,
        "world": world,
        "worker_kwargs": worker_kwargs,
    }

    mp.spawn(_encode_worker_entry, args=(job,), nprocs=world, join=True)


def _encode_worker_entry(rank: int, job: Dict[str, Any]) -> None:
    """
    Worker entry point. Must be top-level for spawn.
    Writes to out_path.part{rank}.
    """
    try:
        mode = job["mode"]
        in_path = job["in_path"]
        out_path = job["out_path"]
        world = job["world"]
        kw = job["worker_kwargs"]

        # Bind rank -> GPU
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpu_id = rank % max(1, n)
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

        out_dir = os.path.dirname(out_path)
        base = os.path.splitext(os.path.basename(out_path))[0]
        part_path = os.path.join(out_dir, f"{base}_shard{rank:02d}.jsonl")

        if mode == "splade":
            _worker_splade(rank, world, device, in_path, part_path, kw)
            return
        if mode == "unicoil":
            _worker_unicoil(rank, world, device, in_path, part_path, kw)
            return
        if mode == "sparta":
            _worker_sparta(rank, world, device, in_path, part_path, kw)
            return
        if mode == "deepct":
            _worker_deepct(rank, world, device, in_path, part_path, kw)
            return
        if mode == "doc2query":
            _worker_doc2query(rank, world, device, in_path, part_path, kw)
            return

        raise ValueError(f"Unknown mode: {mode}")

    except Exception:
        # Persist traceback (SLURM stdout can hide it)
        out_dir = os.path.dirname(job["out_path"])
        log_dir = os.path.join(out_dir, ".worker_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"rank{rank}.log"), "a", encoding="utf-8") as f:
            f.write(f"[WORKER CRASH] rank={rank} mode={job.get('mode')}\n")
            f.write(traceback.format_exc() + "\n")
        raise


# -------------------------
# Model-specific workers
# -------------------------

def _quantize_pairs(pairs, scale: int = 100, min_after_scale: int = 1, topn: int = 256) -> Dict[str, int]:
    items = []
    for tok, w in pairs:
        try:
            w = float(w)
        except Exception:
            continue
        if w <= 0:
            continue
        iw = int(round(w * scale))
        if iw >= min_after_scale:
            items.append((str(tok), iw))

    if topn and len(items) > topn:
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:topn]

    return {t: iw for t, iw in items}


def _worker_splade(rank: int, world: int, device: str, in_path: str, part_path: str, kw: Dict[str, Any]) -> None:
    model_name = kw["model_name"]
    batch_size = int(kw.get("batch_size", 32))
    scale = int(kw.get("scale", 100))
    topn = int(kw.get("topn", 256))

    st_model = STSparseEncoder(model_name, device=device)

    ids: List[str] = []
    texts: List[str] = []

    total_lines = kw.get("total_lines")
    expected = (total_lines + world - 1) // world if total_lines else None

    with open(part_path, "w", encoding="utf-8") as fout:
        for docid, text in _iter_corpus_strided(
            in_path, 
            rank, 
            world,
            expected=expected,
            desc=f"splade rank{rank}",
            position=rank
        ):
            ids.append(docid)
            texts.append(text)

            if len(ids) >= batch_size:
                emb = st_model.encode_document(texts)
                pairs_batch = st_model.decode(emb)

                for did, pairs in zip(ids, pairs_batch):
                    if not pairs:
                        continue
                    vec = _quantize_pairs(pairs, scale=scale, topn=topn)
                    if not vec:
                        continue
                    fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}, ensure_ascii=False) + "\n")

                ids, texts = [], []

        if ids:
            emb = st_model.encode_document(texts)
            pairs_batch = st_model.decode(emb)
            for did, pairs in zip(ids, pairs_batch):
                if not pairs:
                    continue
                vec = _quantize_pairs(pairs, scale=scale, topn=topn)
                if not vec:
                    continue
                fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}, ensure_ascii=False) + "\n")


_PUNC = set(list('.,;:!?()[]{}\'"“”‘’—–-/%'))
_is_bad = re.compile(r'^\W+$').match


def _worker_unicoil(rank: int, world: int, device: str, in_path: str, part_path: str, kw: Dict[str, Any]) -> None:
    # Keep your exact model choice here
    encoder_name = kw.get("encoder_name", "castorini/unicoil-msmarco-passage")
    doc_encoder = UniCoilDocumentEncoder(encoder_name, device=device)

    scale = int(kw.get("scale", 100))
    min_after = int(kw.get("min_after_scale", 1))
    topn = int(kw.get("topn", 500))
    batch_size = int(kw.get("batch_size", 32))

    def normalize(enc_out):
        if isinstance(enc_out, dict):
            return enc_out
        if isinstance(enc_out, list):
            if enc_out and isinstance(enc_out[0], dict):
                return enc_out[0]
            if enc_out and isinstance(enc_out[0], (list, tuple)) and len(enc_out[0]) == 2:
                return {t: float(w) for t, w in enc_out}
        return {}

    def prune_scale(enc_dict):
        items = []
        for tok, w in enc_dict.items():
            try:
                w = float(w)
            except Exception:
                continue
            if w <= 0 or tok in _PUNC or _is_bad(tok):
                continue
            iw = int(round(w * scale))
            if iw >= min_after:
                items.append((tok, iw))
        if topn and len(items) > topn:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:topn]
        return {t: iw for t, iw in items}
    
    ids, texts = [], []

    with open(part_path, "w", encoding="utf-8") as fout:
        for docid, text in _iter_corpus_strided(in_path, rank, world):
            ids.append(docid)
            texts.append(text)

            if len(ids) >= batch_size:
                batch_encodings = doc_encoder.encode(texts)
                for did, enc in zip(ids, batch_encodings):
                    enc = normalize(enc)
                    vec = prune_scale(enc)
                    if vec:
                        fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}, ensure_ascii=False) + "\n")
                ids, texts = [], []


def _worker_sparta(rank: int, world: int, device: str, in_path: str, part_path: str, kw: Dict[str, Any]) -> None:
    model_name = kw["model_name"]
    batch_size = int(kw.get("batch_size", 32))

    sparta = SPARTA(model_name, device=device)

    buf: List[Dict[str, str]] = []
    with open(part_path, "w", encoding="utf-8") as fout:
        for docid, text in _iter_corpus_strided(in_path, rank, world):
            buf.append({"_id": docid, "text": text})
            if len(buf) >= batch_size:
                vecs = sparta.encode_corpus(buf, batch_size=batch_size)  # dict[id] -> dict[token] -> w
                for did, vec in vecs.items():
                    if not vec:
                        continue
                    fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}, ensure_ascii=False) + "\n")
                buf = []

        if buf:
            vecs = sparta.encode_corpus(buf, batch_size=batch_size)
            for did, vec in vecs.items():
                if not vec:
                    continue
                fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}, ensure_ascii=False) + "\n")


def _worker_deepct(rank: int, world: int, device: str, in_path: str, part_path: str, kw: Dict[str, Any]) -> None:
    model_name = kw["model_name"]
    max_seq_len = int(kw.get("max_seq_len", 512)) # DeepCT uses 128; BEIR uses 350
    scale = float(kw.get("scale", 100.0))
    max_tf = int(kw.get("max_tf", 20))
    batch_size = int(kw.get("batch_size", 32))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    model.eval()

    special_tokens = set(tokenizer.all_special_tokens)

    def rewrite_text(texts: List[str]) -> List[str]:
        # Keep behavior consistent with your original per-text stripping/empty handling
        stripped = [t.strip() for t in texts]
        if not stripped:
            return []

        # Batch tokenize + pad
        enc = tokenizer(
            stripped,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits  # [B, L, 1] or [B, L] or [B, L, C]

        input_ids = enc["input_ids"]  # [B, L]
        attn_mask = enc.get("attention_mask", None)

        # Mirror your original `.squeeze(-1)` for the common single-label head case
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # [B, L]

        out: List[str] = []

        for i, text_i in enumerate(stripped):
            if not text_i:
                out.append("")
                continue

            ids_i = input_ids[i]  # [L]
            toks = tokenizer.convert_ids_to_tokens(ids_i)

            if attn_mask is not None:
                valid_len = int(attn_mask[i].sum().item())
                toks = toks[:valid_len]
                scores_iter = logits[i][:valid_len].tolist()
            else:
                scores_iter = logits[i].tolist()

            words: List[str] = []
            scores: List[float] = []
            cur_word = None
            cur_score = None

            for tok, score in zip(toks, scores_iter):
                if tok in special_tokens:
                    continue
                if tok.startswith("##"):
                    sub = tok[2:]
                    if cur_word is None:
                        cur_word, cur_score = sub, score
                    else:
                        cur_word += sub
                else:
                    if cur_word is not None:
                        words.append(cur_word)
                        scores.append(float(cur_score))
                    cur_word, cur_score = tok, score

            if cur_word is not None:
                words.append(cur_word)
                scores.append(float(cur_score))

            rewritten_tokens: List[str] = []
            for w, s in zip(words, scores):
                prob = 1.0 / (1.0 + math.exp(-float(s))) # sigmoid
                tf = int(round(scale * prob)) # scale
                tf = min(tf, max_tf)
                if tf > 0:
                    rewritten_tokens.extend([w] * tf)

            out.append(" ".join(rewritten_tokens) if rewritten_tokens else text_i)

        return out


    with open(part_path, "w", encoding="utf-8") as fout:
        batch_ids, batch_texts = [], []

        for docid, text in _iter_corpus_strided(in_path, rank, world):
            batch_ids.append(docid)
            batch_texts.append(text)

            if len(batch_ids) >= batch_size:
                new_texts = rewrite_text(batch_texts)
                for bid, new_text in zip(batch_ids, new_texts):
                    fout.write(json.dumps({"id": bid, "contents": new_text}, ensure_ascii=False) + "\n")
                batch_ids, batch_texts = [], []

        if batch_ids:
            new_texts = rewrite_text(batch_texts)
            for bid, new_text in zip(batch_ids, new_texts):
                fout.write(json.dumps({"id": bid, "contents": new_text}, ensure_ascii=False) + "\n")


def _worker_doc2query(rank: int, world: int, device: str, in_path: str, part_path: str, kw: Dict[str, Any]) -> None:
    model_ckpt = kw["model_ckpt"]
    queries_per_doc = int(kw.get("queries_per_doc", 3))
    max_input_len = int(kw.get("max_input_len", 320))
    max_query_len = int(kw.get("max_query_len", 64))
    batch_size = int(kw.get("batch_size", 16))
    top_k = int(kw.get("top_k", 10))
    do_sample = bool(kw.get("do_sample", True))

    t5_tokenizer = T5Tokenizer.from_pretrained(model_ckpt)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_ckpt).to(device)
    t5_model.eval()

    ids: List[str] = []
    texts: List[str] = []

    with open(part_path, "w", encoding="utf-8") as fout:
        for docid, text in _iter_corpus_strided(in_path, rank, world):
            ids.append(docid)
            texts.append(text)

            if len(ids) >= batch_size:
                inputs = t5_tokenizer(
                    texts,
                    max_length=max_input_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = t5_model.generate(
                        **inputs,
                        max_length=max_query_len,
                        do_sample=do_sample,
                        top_k=top_k,
                        num_return_sequences=queries_per_doc
                    )
                generated = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i in range(len(texts)):
                    q_slice = generated[i * queries_per_doc:(i + 1) * queries_per_doc]
                    expanded = (texts[i] + " " + " ".join(q_slice)).replace("\t", " ").strip()
                    fout.write(json.dumps({"id": ids[i], "contents": expanded}, ensure_ascii=False) + "\n")

                ids, texts = [], []

        if ids:
            inputs = t5_tokenizer(
                texts,
                max_length=max_input_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = t5_model.generate(
                    **inputs,
                    max_length=max_query_len,
                    do_sample=do_sample,
                    top_k=top_k,
                    num_return_sequences=queries_per_doc
                )
            generated = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i in range(len(texts)):
                q_slice = generated[i * queries_per_doc:(i + 1) * queries_per_doc]
                expanded = (texts[i] + " " + " ".join(q_slice)).replace("\t", " ").strip()
                fout.write(json.dumps({"id": ids[i], "contents": expanded}, ensure_ascii=False) + "\n")


# -------------------------
# Your class (keep init/query encoders)
# -------------------------

class SparseEncoder:
    def __init__(self, model_name, model_key, device):
        self.model_name = model_name
        self.model_key = model_key
        self.device = device.type
        self.query_encoder = None
        if "splade" in model_name:
            self.query_encoder = SpladeQueryEncoder(model_name, device=device)
        if "unicoil" in model_name:
            self.query_encoder = UniCoilQueryEncoder("castorini/unicoil-noexp-msmarco-passage", device=device)
        if "sparta" in model_name:
            self.query_encoder = SPARTA(model_name, device=self.device)

    # Keep your build_d2q_docs if you still need the DataFrame variant
    # (but multi-GPU doc2query corpus encoding now lives in encode() too).

    def encode(self, corpus_dir: str, encoding_dir: str):
        if self.model_name == "bm25":
            return None

        os.makedirs(encoding_dir, exist_ok=True)

        in_path = os.path.join(corpus_dir, "corpus.jsonl")
        out_path = os.path.join(encoding_dir, "corpus.jsonl")

        total_lines = _count_lines(in_path)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return encoding_dir

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; this no-CLI multi-GPU encoder expects GPUs")

        world = min(4, torch.cuda.device_count())

        name = self.model_name.lower()

        if "deepct" in name:
            _spawn_multi_gpu(
                mode="deepct",
                in_path=in_path,
                out_path=out_path,
                world=world,
                worker_kwargs={
                    "model_name": self.model_name,
                    "batch_size": 32,
                    "max_seq_len": 512,
                    "scale": 100.0,
                    "max_tf": 20,
                },
            )
            return encoding_dir

        if "sparta" in name:
            _spawn_multi_gpu(
                mode="sparta",
                in_path=in_path,
                out_path=out_path,
                world=world,
                worker_kwargs={
                    "model_name": self.model_name,
                    "batch_size": 32,
                },
            )
            return encoding_dir

        if "unicoil" in name:
            _spawn_multi_gpu(
                mode="unicoil",
                in_path=in_path,
                out_path=out_path,
                world=world,
                worker_kwargs={
                    "encoder_name": "castorini/unicoil-msmarco-passage",
                    "batch_size": 32,
                    "scale": 100,
                    "min_after_scale": 1,
                    "topn": 500,
                },
            )
            return encoding_dir

        if "splade" in name:
            _spawn_multi_gpu(
                mode="splade",
                in_path=in_path,
                out_path=out_path,
                world=world,
                worker_kwargs={
                    "model_name": self.model_name,
                    "batch_size": 32,
                    "scale": 100,
                    "topn": 256,
                    "total_lines": total_lines,
                },
            )
            return encoding_dir

        if "doc2query" in name:
            from src.configs.models import sparse_models
            model_ckpt = sparse_models["doc2query"]["model_path"]

            # model_ckpt = self.model_name

            _spawn_multi_gpu(
                mode="doc2query",
                in_path=in_path,
                out_path=out_path,
                world=world,
                worker_kwargs={
                    "model_ckpt": model_ckpt,
                    "queries_per_doc": 3,
                    "max_input_len": 320,
                    "max_query_len": 64,
                    "batch_size": 16,
                    "top_k": 10,
                    "do_sample": True,
                },
            )
            return encoding_dir

        # Fallback to your original pyserini.encode command path (single GPU/CPU)
        cmd = [
            sys.executable, "-m", "pyserini.encode",
            "input",  "--corpus", corpus_dir, "--fields", "text",
            "output", "--embeddings", encoding_dir,
            "encoder", "--encoder", self.model_name,
            "--batch", "32", "--device", self.device
        ]
        subprocess.run(cmd, check=True)
        return encoding_dir
