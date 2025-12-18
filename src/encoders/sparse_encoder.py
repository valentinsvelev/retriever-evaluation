import os
import sys
import json
import re
import math
import subprocess
import hashlib
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import torch

from pyserini.encode import (
    SpladeQueryEncoder,
    UniCoilQueryEncoder,
    UniCoilDocumentEncoder,
)
from sentence_transformers import SparseEncoder as STSparseEncoder
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from src.configs.models import MODELS
from src.models.sparta import SPARTA


# -----------------------------------------------------------------------------
# Utility Functions: Hashing & File Operations
# -----------------------------------------------------------------------------
import traceback
def _worker_wrapper(rank, shard_path, part_path, worker_fn, worker_kwargs):
    try:
        worker_fn(rank, shard_path, part_path, worker_kwargs)
    except Exception:
        msg = f"[WORKER CRASH] rank={rank} shard={shard_path} part={part_path}"
        print(msg, flush=True)
        traceback.print_exc()
        # also persist to file
        log_dir = os.path.join(os.path.dirname(part_path), ".worker_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"rank{rank}.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.write(traceback.format_exc() + "\n")
        sys.stderr.flush()
        sys.stdout.flush()
        raise

def _stable_hash(s: str) -> str:
    """Returns a short, deterministic hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _split_jsonl_contiguous(in_path: str, shard_dir: str, num_shards: int) -> List[str]:
    """
    Splits a JSONL file into `num_shards` contiguous chunks physically on disk.
    This prevents seek contention during multiprocessing.
    """
    _ensure_dir(shard_dir)

    # 1. Count total lines quickly
    with open(in_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    num_shards = max(1, min(num_shards, n_lines if n_lines > 0 else 1))
    shard_paths = [os.path.join(shard_dir, f"shard_{i}.jsonl") for i in range(num_shards)]

    # 2. Return existing if valid
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in shard_paths):
        return shard_paths

    # 3. Calculate split sizes
    base = n_lines // num_shards
    rem = n_lines % num_shards
    shard_sizes = [base + (1 if i < rem else 0) for i in range(num_shards)]

    # 4. Write shards
    for p in shard_paths:
        if os.path.exists(p): os.remove(p)

    with open(in_path, "r", encoding="utf-8") as fin:
        for shard_idx, shard_n in enumerate(shard_sizes):
            with open(shard_paths[shard_idx], "w", encoding="utf-8") as fout:
                for _ in range(shard_n):
                    line = fin.readline()
                    if not line: break
                    fout.write(line)

    return shard_paths

def _concat_files_in_order(part_paths: List[str], out_path: str) -> None:
    """Concatenates shard outputs back into a single file in deterministic order."""
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in part_paths:
            if not os.path.exists(p): continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

def _run_on_all_gpus_sharded(
    in_path: str,
    out_path: str,
    num_gpus: int,
    worker_fn,
    worker_kwargs: Optional[Dict[str, Any]] = None,
    shard_cache_root: Optional[str] = None,
) -> None:
    """
    Orchestrator:
    1. Splits input file into N shards.
    2. Spawns N processes (one per GPU).
    3. Runs `worker_fn` on each shard.
    4. Merges outputs.
    """
    worker_kwargs = worker_kwargs or {}
    _ensure_dir(os.path.dirname(out_path))

    # Skip if output already exists
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    # Prepare shards
    if shard_cache_root is None:
        shard_cache_root = os.path.join(os.path.dirname(out_path), ".shards")
    _ensure_dir(shard_cache_root)
    
    # Hash input path to create unique shard directory
    shard_dir = os.path.join(shard_cache_root, f"{_stable_hash(in_path)}_ng{num_gpus}")
    shard_paths = _split_jsonl_contiguous(in_path, shard_dir, num_gpus)

    # Prepare temp output paths
    part_paths = [f"{out_path}.part{i}" for i in range(len(shard_paths))]
    for p in part_paths:
        if os.path.exists(p): os.remove(p)

    # Spawn Workers
    ctx = mp.get_context("spawn")
    procs = []
    for rank, shard_path in enumerate(shard_paths):
        p = ctx.Process(
            target=_worker_wrapper,
            args=(rank, shard_path, part_paths[rank], worker_fn, worker_kwargs),
        )
        p.start()
        procs.append(p)

    # Wait for completion
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker (pid={p.pid}) exited with code {p.exitcode}")

    # Merge and Cleanup
    _concat_files_in_order(part_paths, out_path)
    for p in part_paths:
        try: os.remove(p)
        except OSError: pass


# -----------------------------------------------------------------------------
# Optimized Workers (Batched & Streaming)
# -----------------------------------------------------------------------------

def _deepct_worker(rank: int, shard_in: str, part_out: str, kw: Dict[str, Any]) -> None:
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    
    n = torch.cuda.device_count()
    gpu_id = rank % max(1, n)   # safe even if n==0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model_name = kw["model_name"]
    max_seq_len = kw.get("max_seq_len", 512)
    scale = kw.get("scale", 100.0)
    
    # Batch size for inference (adjust based on GPU VRAM, 64 is usually safe for BERT-base)
    BATCH_SIZE = 64

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()
    special_tokens = set(tokenizer.all_special_tokens)
    MAX_TF = 20

    def process_buffer(buffer_list):
        if not buffer_list: return []
        
        texts = [b["text"] for b in buffer_list]
        
        # 1. Batch Tokenize
        enc = tokenizer(
            texts, truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt"
        )
        input_ids_batch = enc["input_ids"]
        enc = {k: v.to(device) for k, v in enc.items()}

        # 2. Batch Inference
        with torch.no_grad():
            outputs = model(**enc)
            # Logits shape: [batch, seq_len] (squeezed last dim)
            logits_batch = outputs.logits.squeeze(-1).cpu()

        results = []
        
        # 3. Post-process individually
        for i, text in enumerate(texts):
            if not text:
                results.append("")
                continue

            # Slice to actual length
            att_mask = enc["attention_mask"][i].cpu()
            actual_len = att_mask.sum().item()
            
            input_ids = input_ids_batch[i][:actual_len]
            logits = logits_batch[i][:actual_len]
            toks = tokenizer.convert_ids_to_tokens(input_ids)

            words, scores = [], []
            current_word, current_score = None, None

            # DeepCT Aggregation Logic
            for tok, score in zip(toks, logits.tolist()):
                if tok in special_tokens: continue
                if tok.startswith("##"):
                    if current_word: current_word += tok[2:]
                    else: current_word, current_score = tok[2:], score
                else:
                    if current_word:
                        words.append(current_word); scores.append(float(current_score))
                    current_word, current_score = tok, score
            
            if current_word:
                words.append(current_word); scores.append(float(current_score))

            rewritten_tokens = []
            for w, s in zip(words, scores):
                prob = 1.0 / (1.0 + math.exp(-s))
                tf = int(round(scale * prob))
                if tf > 0:
                    rewritten_tokens.extend([w] * min(tf, MAX_TF))

            results.append(" ".join(rewritten_tokens) if rewritten_tokens else text)
        return results

    # Stream processing
    buffer = []
    with open(shard_in, "r", encoding="utf-8") as fin, open(part_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"GPU {rank} DeepCT", position=rank):
            obj = json.loads(line)
            docid = str(obj["id"])
            text = (obj.get("text") or obj.get("contents") or "").strip()
            
            buffer.append({"id": docid, "text": text})

            if len(buffer) >= BATCH_SIZE:
                processed = process_buffer(buffer)
                for item, new_txt in zip(buffer, processed):
                    fout.write(json.dumps({"id": item["id"], "contents": new_txt}, ensure_ascii=False) + "\n")
                buffer = []
        
        if buffer:
            processed = process_buffer(buffer)
            for item, new_txt in zip(buffer, processed):
                fout.write(json.dumps({"id": item["id"], "contents": new_txt}, ensure_ascii=False) + "\n")


def _doc2query_worker(rank: int, shard_in: str, part_out: str, kw: Dict[str, Any]) -> None:
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    
    n = torch.cuda.device_count()
    gpu_id = rank % max(1, n)   # safe even if n==0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    model_ckpt = kw["model_ckpt"]
    queries_per_doc = kw.get("queries_per_doc", 3)
    batch_size = kw.get("batch_size", 16)
    
    t5_tokenizer = T5Tokenizer.from_pretrained(model_ckpt)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_ckpt).to(device).eval()

    buffer = []

    def process_buffer(buf):
        if not buf: return []
        texts = [b["text"] for b in buf]
        
        inputs = t5_tokenizer(
            texts, 
            max_length=kw.get("max_input_len", 320), 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = t5_model.generate(
                **inputs,
                max_length=kw.get("max_query_len", 64),
                do_sample=kw.get("do_sample", True),
                top_k=kw.get("top_k", 10),
                num_return_sequences=queries_per_doc
            )
        
        generated = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        res = []
        for i, item in enumerate(buf):
            orig = item["text"]
            q_slice = generated[i*queries_per_doc : (i+1)*queries_per_doc]
            gen_q = " ".join(q_slice).replace("\t", " ").strip()
            res.append(f"{orig} {gen_q}".strip())
        return res

    with open(shard_in, "r", encoding="utf-8") as fin, open(part_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"GPU {rank} Doc2Query", position=rank):
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("contents") or "").strip()
            if not text: continue
            
            buffer.append({"id": str(obj["id"]), "text": text})

            if len(buffer) >= batch_size:
                expanded_texts = process_buffer(buffer)
                for item, exp in zip(buffer, expanded_texts):
                    fout.write(json.dumps({"id": item["id"], "contents": exp}, ensure_ascii=False) + "\n")
                buffer = []
        
        if buffer:
            expanded_texts = process_buffer(buffer)
            for item, exp in zip(buffer, expanded_texts):
                fout.write(json.dumps({"id": item["id"], "contents": exp}, ensure_ascii=False) + "\n")


def _splade_worker(rank: int, shard_in: str, part_out: str, kw: Dict[str, Any]) -> None:
    n = torch.cuda.device_count()
    gpu_id = rank % max(1, n)   # safe even if n==0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    st_model = STSparseEncoder(kw["model_name"], device=str(device))
    batch_size = kw.get("batch_size", 32)
    buffer = []

    with open(shard_in, "r", encoding="utf-8") as fin, open(part_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"GPU {rank} SPLADE", position=rank):
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("contents") or "").strip()
            if not text: continue
            
            buffer.append({"id": str(obj["id"]), "text": text})
            
            if len(buffer) >= batch_size:
                texts = [b["text"] for b in buffer]
                # encode_document accepts list of strings
                emb = st_model.encode_document(texts)
                pairs_batch = st_model.decode(emb)

                for item, pairs in zip(buffer, pairs_batch):
                    #if not pairs: continue
                    if not pairs:
                        print("EMPTY pairs for doc", item["id"])
                        continue
                    vec = SparseEncoder._quantize_pairs(pairs)
                    fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": item["text"]}, ensure_ascii=False) + "\n")
                buffer = []

        if buffer:
            texts = [b["text"] for b in buffer]
            emb = st_model.encode_document(texts)
            pairs_batch = st_model.decode(emb)
            for item, pairs in zip(buffer, pairs_batch):
                #if not pairs: continue
                if not pairs:
                    print("EMPTY pairs for doc", item["id"])
                    continue
                vec = SparseEncoder._quantize_pairs(pairs)
                fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": item["text"]}, ensure_ascii=False) + "\n")


def _unicoil_worker(rank: int, shard_in: str, part_out: str, kw: Dict[str, Any]) -> None:
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    
    n = torch.cuda.device_count()
    gpu_id = rank % max(1, n)   # safe even if n==0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    doc_encoder = UniCoilDocumentEncoder("castorini/unicoil-msmarco-passage", device=str(device))
    
    # Pruning Constants
    SCALE = 100
    _PUNC = set(list('.,;:!?()[]{}\'"“”‘’—–-/%'))

    def prune(enc_dict):
        items = []
        for t, w in enc_dict.items():
            try: w = float(w)
            except: continue
            if w > 0 and t not in _PUNC:
                val = int(round(w * SCALE))
                if val >= 1: items.append((t, val))
        return dict(items)

    batch_size = 64
    buffer = []

    with open(shard_in, "r", encoding="utf-8") as fin, open(part_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"GPU {rank} UniCoil", position=rank):
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("contents") or "").strip()
            if not text: continue
            
            buffer.append({"id": str(obj["id"]), "text": text})

            if len(buffer) >= batch_size:
                texts = [b["text"] for b in buffer]
                
                # Attempt batch encoding
                try:
                    batch_encs = doc_encoder.encode(texts)
                    # Handle single-item return
                    if isinstance(batch_encs, dict): batch_encs = [batch_encs]
                except:
                    # Fallback
                    batch_encs = [doc_encoder.encode(t) for t in texts]

                for item, enc in zip(buffer, batch_encs):
                    if isinstance(enc, list): enc = {t: float(w) for t, w in enc}
                    vec = prune(enc)
                    if vec:
                        #fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": ""}) + "\n")
                        fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": item["text"]}, ensure_ascii=False) + "\n")
                buffer = []

        if buffer:
            texts = [b["text"] for b in buffer]
            batch_encs = [doc_encoder.encode(t) for t in texts]
            for item, enc in zip(buffer, batch_encs):
                if isinstance(enc, list): enc = {t: float(w) for t, w in enc}
                vec = prune(enc)
                if vec:
                    #fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": ""}) + "\n")
                    fout.write(json.dumps({"id": item["id"], "vector": vec, "contents": item["text"]}, ensure_ascii=False) + "\n")


def _sparta_worker(rank: int, shard_in: str, part_out: str, kw: Dict[str, Any]) -> None:
    # torch.cuda.set_device(rank)
    # device = torch.device(f"cuda:{rank}")
    
    n = torch.cuda.device_count()
    gpu_id = rank % max(1, n)   # safe even if n==0
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    model_name = kw["model_name"]
    sparta = SPARTA(model_name, device=str(device))
    
    # Large chunk size to feed SPARTA's internal batcher, but not entire file
    CHUNK_SIZE = 5000 
    buffer = []

    with open(shard_in, "r", encoding="utf-8") as fin, open(part_out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"GPU {rank} SPARTA", position=rank):
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("contents") or "").strip()
            if not text: continue
            
            buffer.append({"_id": str(obj["id"]), "text": text})

            if len(buffer) >= CHUNK_SIZE:
                id2text = {b["_id"]: b["text"] for b in buffer}
                
                # encode_corpus handles internal batching
                vecs = sparta.encode_corpus(buffer, batch_size=kw.get("batch_size", 16))
                for did, vec in vecs.items():
                    if vec:
                        fout.write(json.dumps({"id": did, "vector": vec, "contents": id2text[did]}) + "\n")
                buffer = []
        
        if buffer:
            id2text = {b["_id"]: b["text"] for b in buffer}
            vecs = sparta.encode_corpus(buffer, batch_size=kw.get("batch_size", 16))
            for did, vec in vecs.items():
                if vec:
                    fout.write(json.dumps({"id": did, "vector": vec, "contents": id2text[did]}) + "\n")


# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------

class SparseEncoder:
    def __init__(self, model_name, model_key, device):
        self.model_name = model_name
        self.model_key = model_key

        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device) if isinstance(device, str) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_gpu = torch.cuda.device_count()

        # Instantiate query encoders for immediate use if needed (CPU side)
        self.query_encoder = None
        if "splade" in model_name:
            self.query_encoder = SpladeQueryEncoder(model_name, device=self.device)
        if "unicoil" in model_name:
            self.query_encoder = UniCoilQueryEncoder("castorini/unicoil-noexp-msmarco-passage", device=self.device)
        if "sparta" in model_name:
            self.query_encoder = SPARTA(model_name, device=str(self.device))

    @staticmethod
    def _quantize_pairs(pairs):
        SCALE = 100
        MIN_AFTER_SCALE = 1
        TOPN = 256
        items = []
        for tok, w in pairs:
            w = float(w)
            if w <= 0: continue
            iw = int(round(w * SCALE))
            if iw >= MIN_AFTER_SCALE: items.append((tok, iw))
        
        if TOPN and len(items) > TOPN:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:TOPN]
        return {t: iw for t, iw in items}

    def _deepct_rewrite_corpus(self, corpus_dir: str, encoding_dir: str, max_seq_len: int = 512, scale: float = 100.0) -> str:
        in_path = os.path.join(corpus_dir, "corpus.jsonl")
        out_path = os.path.join(encoding_dir, "corpus.jsonl")
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[DeepCT] Output exists: {out_path}")
            return encoding_dir

        # Use Sharded Multiprocessing if > 1 GPU
        num_workers = self.n_gpu if self.n_gpu > 1 else 1
        print(f"[DeepCT] Running with {num_workers} workers.")
        
        _run_on_all_gpus_sharded(
            in_path=in_path,
            out_path=out_path,
            num_gpus=num_workers,
            worker_fn=_deepct_worker,
            worker_kwargs={"model_name": self.model_name, "max_seq_len": max_seq_len, "scale": scale},
            shard_cache_root=os.path.join(encoding_dir, ".shards"),
        )
        return encoding_dir

    def build_d2q_docs(self, docs_df: pd.DataFrame, out_dir: str, model_ckpt: str, **kwargs) -> str:
        print("\n--- Expanding Corpus with doc2query ---")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "corpus.jsonl")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_dir

        # Write temp input
        tmp_in = os.path.join(out_dir, "doc2query_input.jsonl")
        if not (os.path.exists(tmp_in) and os.path.getsize(tmp_in) > 0):
            with open(tmp_in, "w", encoding="utf-8") as f:
                for did, text in zip(docs_df["doc_id"].astype(str), docs_df["text"].astype(str)):
                    f.write(json.dumps({"id": did, "contents": text}, ensure_ascii=False) + "\n")

        num_workers = self.n_gpu if self.n_gpu > 1 else 1
        print(f"[doc2query] Running with {num_workers} workers.")
        
        _run_on_all_gpus_sharded(
            in_path=tmp_in,
            out_path=out_path,
            num_gpus=num_workers,
            worker_fn=_doc2query_worker,
            worker_kwargs={"model_ckpt": model_ckpt, **kwargs},
            shard_cache_root=os.path.join(out_dir, ".shards"),
        )
        return out_dir

    def encode(self, corpus_dir: str, encoding_dir: str):
        if self.model_name == "bm25": return None
        os.makedirs(encoding_dir, exist_ok=True)

        if "deepct" in self.model_name.lower():
            return self._deepct_rewrite_corpus(corpus_dir, encoding_dir)

        in_path = os.path.join(corpus_dir, "corpus.jsonl")
        out_path = os.path.join(encoding_dir, "corpus.jsonl")
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Encodings exist at {out_path}")
            return encoding_dir

        num_workers = self.n_gpu if self.n_gpu > 1 else 1
        print(f"[Encode] Model: {self.model_name} | Workers: {num_workers}")

        if "sparta" in self.model_name.lower():
            _run_on_all_gpus_sharded(
                in_path, out_path, num_workers, _sparta_worker,
                worker_kwargs={"model_name": self.model_name},
                shard_cache_root=os.path.join(encoding_dir, ".shards")
            )
            return encoding_dir

        if "unicoil" in self.model_name.lower():
            _run_on_all_gpus_sharded(
                in_path, out_path, num_workers, _unicoil_worker,
                worker_kwargs={},
                shard_cache_root=os.path.join(encoding_dir, ".shards")
            )
            return encoding_dir

        if "splade" in self.model_name.lower():
            _run_on_all_gpus_sharded(
                in_path, out_path, num_workers, _splade_worker,
                worker_kwargs={"model_name": self.model_name},
                shard_cache_root=os.path.join(encoding_dir, ".shards")
            )
            if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
                raise RuntimeError(f"[SPLADE] Encoding failed: missing/empty {out_path}")
            return encoding_dir

        # Fallback CLI
        cmd = [
            sys.executable, "-m", "pyserini.encode",
            "input", "--corpus", corpus_dir, "--fields", "text",
            "output", "--embeddings", encoding_dir,
            "encoder", "--encoder", self.model_name,
            "--batch", "32", "--device", str(self.device),
        ]
        subprocess.run(cmd, check=True)
        return encoding_dir
    