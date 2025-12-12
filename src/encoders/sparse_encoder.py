import os
import sys
import json
import re
import subprocess
import pandas as pd
from tqdm import tqdm
import torch
from pyserini.encode import (
    SpladeQueryEncoder,
    UniCoilQueryEncoder,
    UniCoilDocumentEncoder,
)
from sentence_transformers import SparseEncoder as STSparseEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification

from src.configs.models import MODELS
from src.models.sparta import SPARTA


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

    def _deepct_rewrite_corpus(self, corpus_dir: str, encoding_dir: str,
                               max_seq_len: int = 512,
                               scale: float = 100.0) -> str:
        """
        DeepCT-Index style rewriting:
        - Use macavaney/deepct (regression per token)
        - Aggregate to word-level
        - Pass scores through sigmoid to [0,1]
        - Scale, round to int, cap tf
        - Repeat words according to integer tf and index with BM25

        Reads:  corpus_dir/corpus.jsonl
        Writes: encoding_dir/corpus.jsonl
        """
        import math

        in_path = os.path.join(corpus_dir, "corpus.jsonl")
        out_path = os.path.join(encoding_dir, "corpus.jsonl")
        os.makedirs(encoding_dir, exist_ok=True)

        # Reuse if already there
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[DeepCT] Rewritten corpus already exists at {out_path}, reusing.")
            return encoding_dir

        print(f"[DeepCT] Loading DeepCT model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        special_tokens = set(tokenizer.all_special_tokens)
        MAX_TF = 20

        def rewrite_text(text: str) -> str:
            text = text.strip()
            if not text:
                return ""

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                # assume shape [1, seq_len, 1]
                logits = outputs.logits.squeeze(-1).squeeze(0)  # [seq_len]

            input_ids = enc["input_ids"].squeeze(0)
            toks = tokenizer.convert_ids_to_tokens(input_ids)

            # Aggregate scores at word level (merge wordpieces)
            words: list[str] = []
            scores: list[float] = []

            current_word = None
            current_score = None

            for tok, score in zip(toks, logits.tolist()):
                if tok in special_tokens:
                    continue

                if tok.startswith("##"):
                    # continuation of previous word
                    sub = tok[2:]
                    if current_word is None:
                        current_word = sub
                        current_score = score
                    else:
                        current_word += sub
                        # keep the existing score (first-piece score)
                else:
                    # start of a new word -> commit previous
                    if current_word is not None:
                        words.append(current_word)
                        scores.append(float(current_score))
                    current_word = tok
                    current_score = score

            # commit last word
            if current_word is not None:
                words.append(current_word)
                scores.append(float(current_score))

            # Convert scores -> integer term frequencies (DeepCT-ish)
            rewritten_tokens = []
            for w, s in zip(words, scores):
                # sigmoid squashing to [0,1]
                raw = float(s)
                prob = 1.0 / (1.0 + math.exp(-raw))     # in [0,1]
                s_clipped = max(0.0, prob)              # redundant but explicit
                tf = int(round(scale * s_clipped))      # e.g. 100 * prob
                tf = min(tf, MAX_TF)                    # cap TF
                if tf > 0:
                    rewritten_tokens.extend([w] * tf)

            # Fallback: if everything dropped, use original text
            if not rewritten_tokens:
                return text

            return " ".join(rewritten_tokens)

        written = 0
        from tqdm import tqdm
        import json

        with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="[DeepCT] Rewriting corpus"):
                obj = json.loads(line)
                docid = str(obj["id"])
                text = (obj.get("text") or obj.get("contents") or "").strip()
                if not text:
                    continue
                new_text = rewrite_text(text)
                fout.write(json.dumps({"id": docid, "contents": new_text}, ensure_ascii=False) + "\n")
                written += 1

        print(f"[DeepCT] Rewritten {written} documents -> {out_path}")
        return encoding_dir


    def _quantize_pairs(self, pairs):
        """
        pairs: list[(token, weight)]
        -> dict[token] -> int impact
        """
        
        SCALE = 100
        MIN_AFTER_SCALE = 1
        TOPN = 256
        
        items = []
        for tok, w in pairs:
            w = float(w)
            if w <= 0:
                continue
            iw = int(round(w * SCALE))
            if iw >= MIN_AFTER_SCALE:
                items.append((tok, iw))

        if TOPN and len(items) > TOPN:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:TOPN]

        return {t: iw for t, iw in items}


    def build_d2q_docs(
        self,
        docs_df: pd.DataFrame,
        out_dir: str,
        model_ckpt: str,
        queries_per_doc: int = 3,
        max_input_len: int = 320,
        max_query_len: int = 64,
        batch_size: int = 16,
        top_k: int = 10,
        do_sample: bool = True,
    ) -> str:
        """
        Expand documents with docT5query and write a Pyserini JSONL corpus
        """
        print("\n--- Expanding Corpus with doc2query ---")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "corpus.jsonl")

        # Cache: if we already expanded, just reuse it
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Found existing expanded corpus at '{out_path}'.")
            return out_dir

        # Load T5
        print(f"Loading docT5query checkpoint: {model_ckpt}")
        t5_tokenizer = T5Tokenizer.from_pretrained(model_ckpt)
        t5_model = T5ForConditionalGeneration.from_pretrained(model_ckpt).to(self.device)

        # Pull IDs/texts from your DataFrame
        doc_ids   = docs_df["doc_id"].astype(str).tolist()
        doc_texts = docs_df["text"].astype(str).tolist()

        def chunked(lst, n):
            for i in range(0, len(lst), n):
                yield i, lst[i:i+n]

        written = 0
        with open(out_path, "w", encoding="utf-8") as f_out:
            for start, texts in tqdm(list(chunked(doc_texts, batch_size)), desc="Expanding Documents"):
                inputs = t5_tokenizer(
                    texts,
                    max_length=max_input_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = t5_model.generate(
                    **inputs,
                    max_length=max_query_len,
                    do_sample=do_sample,
                    top_k=top_k,
                    num_return_sequences=queries_per_doc
                )
                generated = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i in range(len(texts)):
                    did = doc_ids[start + i]
                    orig = texts[i]
                    # take the slice of size queries_per_doc for this doc
                    q_slice = generated[i*queries_per_doc : (i+1)*queries_per_doc]
                    gen_q   = " ".join(q_slice).replace("\t", " ").strip()
                    expanded_text = f"{orig} {gen_q}".strip()
                    f_out.write(json.dumps({"id": did, "contents": expanded_text}, ensure_ascii=False) + "\n")
                    written += 1

        print(f"Expanded {written} documents -> {out_path}")
        return out_dir


    def encode(self, corpus_dir: str, encoding_dir: str):
        if self.model_name == "bm25":
            return None

        os.makedirs(encoding_dir, exist_ok=True)

        if "deepct" in self.model_name.lower():
            return self._deepct_rewrite_corpus(corpus_dir=corpus_dir, encoding_dir=encoding_dir)


        if "sparta" in self.model_name.lower():
            in_path  = os.path.join(corpus_dir, "corpus.jsonl")
            out_path = os.path.join(encoding_dir, "corpus.jsonl")

            # Build a minimal corpus list: [{"_id": str, "text": str}]
            corpus_list = []
            with open(in_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line)
                    did = str(obj["id"])
                    text = (obj.get("text") or obj.get("contents") or "").strip()
                    if not text:
                        continue
                    corpus_list.append({"_id": did, "text": text})

            # Encode
            sparta = SPARTA(self.model_name, device=self.device)
            sparse_vectors = sparta.encode_corpus(corpus_list, batch_size=4)

            # Write JsonVectorCollection format expected by Pyserini impact indexer
            written = 0
            with open(out_path, "w", encoding="utf-8") as fout:
                for did, vec in sparse_vectors.items():
                    if not vec:
                        continue
                    fout.write(json.dumps({"id": did, "vector": vec, "contents": ""}) + "\n") # include empty "contents" to satisfy DefaultLuceneDocumentGenerator
                    written += 1

            print(f"SPARTA impact encodings written: {written} -> {out_path}")
            return encoding_dir


        if "unicoil" in self.model_name:
            in_path  = os.path.join(corpus_dir, "corpus.jsonl")
            out_path = os.path.join(encoding_dir, "corpus.jsonl")

            doc_encoder = UniCoilDocumentEncoder("castorini/unicoil-msmarco-passage", device=self.device)

            SCALE = 100
            MIN_AFTER_SCALE = 1
            TOPN = 500
            _PUNC = set(list('.,;:!?()[]{}\'"“”‘’—–-/%'))
            _is_bad = re.compile(r'^\W+$').match

            def normalize(enc_out):
                # Return a dict[token]->float
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
                    iw = int(round(w * SCALE))
                    if iw >= MIN_AFTER_SCALE:
                        items.append((tok, iw))
                if TOPN and len(items) > TOPN:
                    items.sort(key=lambda x: x[1], reverse=True)
                    items = items[:TOPN]
                return {t: iw for t, iw in items}

            written = 0
            with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    obj = json.loads(line)
                    docid = str(obj["id"])
                    text = (obj.get("text") or obj.get("contents") or "").strip()
                    if not text:
                        continue
                    enc = doc_encoder.encode(text)          # may be dict OR list
                    enc = normalize(enc)                    # <-- make it a dict
                    vec = prune_scale(enc)
                    if not vec:
                        continue
                    # include empty contents to avoid NPE in DefaultLuceneDocumentGenerator
                    fout.write(json.dumps({"id": docid, "vector": vec, "contents": ""}) + "\n")
                    written += 1

            print(f"uniCOIL impact encodings written: {written} -> {out_path}")
            return encoding_dir
        
        if "splade" in self.model_name.lower():
            in_path  = os.path.join(corpus_dir, "corpus.jsonl")
            out_path = os.path.join(encoding_dir, "corpus.jsonl")

            st_model = STSparseEncoder(self.model_name, device=self.device)

            written = 0
            with open(in_path, "r", encoding="utf-8") as fin, \
                 open(out_path, "w", encoding="utf-8") as fout:

                texts, ids = [], []
                for line in fin:
                    obj = json.loads(line)
                    ids.append(str(obj["id"]))
                    texts.append((obj.get("text") or obj.get("contents") or "").strip())

                BATCH_SIZE = 32
                for i in range(0, len(texts), BATCH_SIZE):
                    chunk = texts[i:i+BATCH_SIZE]

                    emb = st_model.encode_document(chunk) # sparse tensor
                    pairs_per_doc = st_model.decode(emb)

                    for j, pairs in enumerate(pairs_per_doc):
                        vec = self._quantize_pairs(pairs) # prune some weights
                        
                        if not pairs:
                            continue

                        rec = {
                            "id": ids[i + j],
                            "vector": vec,
                            "contents": "" # empty contents so DefaultLuceneDocumentGenerator doesn't NPE
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1

            print(f"SPLADE encodings written (STSparseEncoder, no scaling/filter/pruning): {written} -> {out_path}")
            return encoding_dir

        cmd = [
            sys.executable, "-m", "pyserini.encode",
            "input",  "--corpus", corpus_dir, "--fields", "text",
            "output", "--embeddings", encoding_dir,
            "encoder","--encoder", self.model_name,
                      "--batch", "32", "--device", self.device
        ]
        subprocess.run(cmd, check=True)

        print(f"Encoded documents written to: {encoding_dir}")
        return encoding_dir