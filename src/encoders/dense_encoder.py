import os
import sys
import subprocess
import json
from tqdm import tqdm
import torch
from torch.amp import autocast
import numpy as np
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig,
    T5Tokenizer, T5ForConditionalGeneration,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoModelForCausalLM, T5EncoderModel
)
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel, FlagModel
from InstructorEmbedding import INSTRUCTOR
from gritlm import GritLM
from peft import PeftModel, PeftConfig

if not os.path.exists("tart"):
    subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/tart.git"])
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "tart/TART/requirements.txt"])

from tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer


def _has_tokenizer_sep(tokenizer):
    return getattr(tokenizer, "sep_token", None) is not None

def _join_with_sep(tokenizer, instr: str, text: str) -> str:
    if instr is None or instr == "":
        return text
    if _has_tokenizer_sep(tokenizer):
        return f"{instr} {tokenizer.sep_token} {text}"
    return f"{instr} [SEP] {text}"

def generate_docs_for_query_expansion(query_ids: list, query_texts: list, model_name: str, device: str, path: str, batch_size: int = 64):
    """Generate documents for HyDE and query2doc using Qwen2.5-7B-Instruct."""

    # Load cached output if available
    if os.path.exists(path):
        print(f"Loading pre-generated documents from '{path}'.")
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached

    # Load Qwen2.5 model
    print(f"Loading model for HyDE generation: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )#.to(device)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Build chat-style prompts
    def build_messages(query: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-focused assistant. "
                    "Given a query, you write a *self-contained*, factual passage that could appear in a document relevant to that query.\n\n"
                    "Requirements:\n"
                    "- Do NOT mention that you are an assistant or that this is hypothetical.\n"
                    "- Do NOT restate the query.\n"
                    "- No bullet points or headings.\n"
                    "- Write one coherent paragraph of 150–300 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    "Write a single, detailed passage that could answer this query. "
                    "Return only the passage text."
                ),
            },
        ]

    all_outputs = []

    # Generation loop
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Generating docs with Qwen2.5"):
        batch_queries = query_texts[i:i + batch_size]

        # Build chat templates
        batch_texts = [
            tokenizer.apply_chat_template(
                build_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_queries
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )#.to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                #top_p=0.95,
                #repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt portion → keep only new generation
        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # Light post-processing: strip whitespace
        decoded = [d.strip() for d in decoded]
        all_outputs.extend(decoded)

    # Save & return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f)

    print(f"Hypothetical documents saved to '{path}'.")
    return all_outputs


def generate_pseudo_docs_for_query_expansion(query_ids: list, query_texts: list, model_name: str, device: str, path: str, batch_size: int = 64):
    """Generate documents for Query2doc and query2doc using Qwen2.5-7B-Instruct."""

    # Load cached output if available
    if os.path.exists(path):
        print(f"Loading pre-generated documents from '{path}'.")
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached

    # Load Qwen2.5 model
    print(f"Loading model for Query2doc generation: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Build chat-style prompts
    def build_messages(query: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-focused assistant. "
                    "Write a passage that answers the given query:.\n\n"
                    "Requirements:\n"
                    "- Do NOT mention that you are an assistant or that this is hypothetical.\n"
                    "- Do NOT restate the query.\n"
                    "- No bullet points or headings.\n"
                    "- Write one coherent paragraph of 150–300 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    "Write a single, detailed passage that could answer this query. "
                    "Return only the passage text."
                ),
            },
        ]

    #all_outputs = []
    out_dict = {}

    # Generation loop
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Generating docs with Qwen2.5"):
        batch_qids = query_ids[i:i + batch_size]
        batch_queries = query_texts[i:i + batch_size]

        # Build chat templates
        batch_texts = [
            tokenizer.apply_chat_template(
                build_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_queries
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )#.to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=1,
                #top_p=0.95,
                #repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt portion → keep only new generation
        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]
        # all_outputs.extend(decoded)

        for qid, q, pseudo in zip(batch_qids, batch_queries, decoded):
            out_dict[str(qid)] = {"query": q, "pseudo_doc": pseudo}

    # Save & return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False)

    print(f"Hypothetical documents saved to '{path}'.")
    return out_dict #all_outputs


class DenseEncoder:
    """Handles loading and encoding text for various model types, and TART-full reranking."""
    def __init__(self, model_name, config, device):
        self.model_name = model_name
        self.config = config or {}
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.multi_gpu = (
            torch.cuda.is_available()
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        )
        if self.multi_gpu:
            print(f"Detected {torch.cuda.device_count()} GPUs - will use DataParallel where possible.")
        
        self.bnb_cfg_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        self.bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        self._load_model()
        
        self._dual_family = None

    # -------------------- LOADING --------------------
    def _load_model(self):
        if isinstance(self.model_name, dict):
            self._handler_type = 'dual_encoder'
            print("Loading with: Dual Encoder Handler")
            self._load_dual_encoder()
            return

        name = str(self.model_name).lower()

        if "tart" in name:
            self._handler_type = 'tart_reranker'
            print("Loading with: TART-full Reranker Handler")
            self._load_tart_reranker()
            return

        if "bge" in name:
            self._handler_type = 'flag_embedding'
            print("Loading with: FlagEmbedding Handler")
            self.model = FlagModel(self.model_name, query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
            return

        # if "instructor" in name:
        #     self._handler_type = 'instructor'
        #     print("Loading with: INSTRUCTOR Handler")
        #     device_str = self.device.type
        #     self.model = INSTRUCTOR(self.model_name)
            
        #     # multi-GPU: start pool if we have >1 GPU
        #     self.pool = None
        #     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #         num_gpus = min(4, torch.cuda.device_count())
        #         target_devices = [f"cuda:{i}" for i in range(num_gpus)]
        #         print(f"Starting SentenceTransformer multi-process pool on {target_devices}")
        #         self.pool = self.model.start_multi_process_pool(
        #             target_devices=target_devices
        #         )
            
        #     return

        if "promptriever" in name or "repllama" in name:
            self._handler_type = 'peft_biencoder'
            print("Loading with: PEFT Bi-Encoder Handler (Promptriever/RepLLaMA)")
            self._load_peft_biencoder()
            return
        
        if "gritlm" in name:
            self._handler_type = 'gritlm'
            print("Loading with: GritLM Handler")
            self.model = GritLM(self.model_name, device_map="auto", mode="embedding", torch_dtype=torch.float16)
            self.model.model.config.use_cache = False
            return

        if "sentence-transformers" in name or "gtr-" in name or "gemma" in name or "e5-mistral" in name or "kalm" in name or "instructor" in name:
            self._handler_type = 'sentence_transformer'
            print("Loading with: SentenceTransformer Handler")
            
            # Quantization config
            if "qwen3" in name or "e5-mistral" in name:
                model_kwargs = {
                    "trust_remote_code": True,
                    #"quantization_config": self.bnb_cfg,
                    "load_in_4bit": True
                }
                print("Quantization applied")
            else:
                model_kwargs={"torch_dtype": torch.bfloat16}

            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                #model_kwargs=model_kwargs
                #device=self.device
            )

            if "kalm" in name:
                self.model.max_seq_length = 512
            
            # multi-GPU: start pool if we have >1 GPU
            self.pool = None
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                num_gpus = min(4, torch.cuda.device_count())
                target_devices = [f"cuda:{i}" for i in range(num_gpus)]
                print(f"Starting SentenceTransformer multi-process pool on {target_devices}")
                self.pool = self.model.start_multi_process_pool(
                    target_devices=target_devices
                )

            if "nvembed" in name:
                self.model.max_seq_length = 32768
                self.model.tokenizer.padding_side="right"

            return

        # Generic loader
        self._handler_type = 'transformer'
        print("Loading with: Manual Transformers Handler")
        self._load_manual_transformer()

    def _load_manual_transformer(self):
        name = str(self.model_name).lower()

        # You can also toggle this via self.config["quantize"] = True/False
        want_quant = self.config.get("quantize")
        if want_quant is None:
            want_quant = (self.device.type == "cuda" and any(x in name for x in ["4b", "7b", "8b", "xl", "xxl", "gritlm", "sfr"]))
        
        # config = AutoConfig.from_pretrained(name, trust_remote_code=True)

        if want_quant:
            print("Loading quantized model.")
            gpu_idx = (self.device.index if self.device.index is not None else 0)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=self.bnb_cfg,
                device_map={"": gpu_idx},
                torch_dtype=None,
                low_cpu_mem_usage=True,
            )
            if "gte" in name or "nv-embed" in name:
                print("Applying use_cache = False")
                self.model.config.use_cache = False
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)
            
            if self.multi_gpu:
                print("[DenseEncoder] Wrapping transformer model in DataParallel")
                self.model = torch.nn.DataParallel(self.model)

        if "repllama" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", use_fast=True, trust_remote_code=True, clean_up_tokenization_spaces=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, trust_remote_code=True, clean_up_tokenization_spaces=True
            )

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"

        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = True

        # Debug: confirm what we actually loaded
        try:
            print("quantization_method:", getattr(self.model, "quantization_method", None))
            print("hf_device_map:", getattr(self.model, "hf_device_map", None))
        except Exception:
            pass

    def _load_peft_biencoder(self):
        peft_id = self.model_name
        peft_cfg = PeftConfig.from_pretrained(peft_id)
        base_id = peft_cfg.base_model_name_or_path
        
        #gpu_idx = (self.device.index if self.device.index is not None else 0)
        base = AutoModel.from_pretrained(
            base_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Load and adjust tokenizer
        if "repllama" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
        
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "right"

        model = PeftModel.from_pretrained(base, peft_id)
        model = model.merge_and_unload()
        
        # if "promptriever" in self.model_name:
        model.config.max_length = 512
        tokenizer.model_max_length = 512
        # elif "repllama" in self.model_name:
        #     model.config.max_length = 2048
        #     tokenizer.model_max_length = 2048

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

    def _load_dual_encoder(self):
        q_name = self.model_name.get("question")
        c_name = self.model_name.get("context")
        if ("dpr" in q_name.lower()) or ("dpr" in c_name.lower()):
            self._dual_family = "dpr"
            self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_name, use_fast=True)
            base_q = DPRQuestionEncoder.from_pretrained(q_name)
            self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(c_name, use_fast=True)
            base_c = DPRContextEncoder.from_pretrained(c_name)
            
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"Using DataParallel for DPR on {torch.cuda.device_count()} GPUs")
                self.question_model = torch.nn.DataParallel(base_q).cuda()
                self.context_model  = torch.nn.DataParallel(base_c).cuda()
            else:
                self.question_model = base_q.to(self.device)
                self.context_model  = base_c.to(self.device)
        else:
            self._dual_family = "generic"
            self.question_tokenizer = AutoTokenizer.from_pretrained(q_name, use_fast=True)
            self.question_model = AutoModel.from_pretrained(q_name, device_map="auto")
            self.context_tokenizer = AutoTokenizer.from_pretrained(c_name, use_fast=True)
            self.context_model = AutoModel.from_pretrained(c_name, device_map="auto")

    def _load_tart_reranker(self):
        self.tokenizer = EncT5Tokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = EncT5ForSequenceClassification.from_pretrained(self.model_name, device_map="auto")

    # ----------------------------------------------------------------
    # -------------------- POOLING & INSTRUCTIONS --------------------
    # ----------------------------------------------------------------
    def _pooling(self, outputs, attention_mask, strategy):
        last_hidden = outputs.last_hidden_state
        if strategy == 'cls':
            name = str(self.model_name).lower()
            if "simcse" in name and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return last_hidden[:, 0, :]
        if strategy == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * input_mask_expanded, 1)
            denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return summed / denom
        if strategy == 'last_token':
            idxs = attention_mask.sum(dim=1).long() - 1
            return last_hidden[torch.arange(last_hidden.size(0)), idxs]

    def _apply_instruction(self, batch_texts, is_query: bool):
        instruction = self.config.get('query_instruction') if is_query else self.config.get('doc_instruction')
        if not instruction:
            return batch_texts
        if "{}" in instruction:
            return [instruction.format(text) for text in batch_texts]
        return [f'{instruction}{text}' for text in batch_texts]
    
    def _gritlm_instruction(self, instruction: str, is_query: bool) -> str:
        # Docs → no user turn, just the embedding tag
        if not is_query:
            return "<|embed|>\n"
        # Queries → user turn + instruction + embed tag
        if instruction:
            return f"<|user|>\n{instruction}\n<|embed|>\n"
        else:
            return "<|embed|>\n"

    def _nv_instruction(self, instruction: str, is_query: bool) -> str:
        if not is_query:
            return ""
        instr = instruction or "Given a question, retrieve passages that answer the question"
        return f"Instruct: {instr}\nQuery: "

    # --------------------------------------------------
    # -------------------- ENCODING --------------------
    # --------------------------------------------------
    def encode(self, texts, titles=None, is_query=False, batch_size=256):
        """Encodes a list of texts into embeddings (except for TART-full, which is a reranker)."""
        
        if self._handler_type == "gritlm":
            batch_size = 8
        # elif self._handler_type == "instructor":
        #     batch_size = 64
        elif self._handler_type == "peft_biencoder":
            if "repllama" in self.model_name:
                batch_size = 8
            else:
                batch_size = 64 # Promptriever
        elif "gtr" in self.model_name:
            batch_size = 128
        elif "jina" in self.model_name:
            batch_size = 256
        elif self._handler_type == "transformer":
            batch_size = 256
        elif self._handler_type == "sentence_transformer":
            batch_size = 32

        print(f"Using a batch size of {batch_size}.")
        
        # FlagEmbedding (BGE)
        if self._handler_type == 'flag_embedding':
            params = {"batch_size": batch_size}
            embs = self.model.encode(texts, **params)
            return torch.from_numpy(embs)

        # Sentence-Transformers
        if self._handler_type == 'sentence_transformer':
            texts_to_encode = texts
            if not is_query and titles is not None and len(texts) == len(titles):
                print("Concatenating titles and texts for document encoding.")
                texts_to_encode = [f"{title} {text}" for title, text in zip(titles, texts)]
            embeddings = self.model.encode(texts_to_encode, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, pool=self.pool)
            return torch.from_numpy(embeddings)

        # # INSTRUCTOR
        # if self._handler_type == 'instructor':
        #     instr = self.config.get('query_instruction') if is_query else self.config.get('doc_instruction')
        #     pairs = [[instr, t] for t in texts]
        #     embs = self.model.encode(pairs, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        #     return torch.from_numpy(np.asarray(embs))

        # GritLM
        if self._handler_type == 'gritlm':
            # concat titles for docs (keeps your existing behavior)
            texts_to_encode = texts
            if not is_query and titles is not None and len(texts) == len(titles):
                print("Concatenating titles and texts for document encoding.")
                texts_to_encode = [f"{title} {text}" for title, text in zip(titles, texts)]

            instr_cfg = self.config.get('query_instruction') if is_query else self.config.get('doc_instruction')
            grit_instr = self._gritlm_instruction(instr_cfg or "", is_query=is_query)

            embs = self.model.encode(texts_to_encode, instruction=grit_instr, batch_size=batch_size)

            # Convert to torch and (optionally) normalize to match other handlers
            if isinstance(embs, np.ndarray):
                embs = torch.from_numpy(embs)
            elif isinstance(embs, list):
                embs = torch.from_numpy(np.asarray(embs))
            elif not torch.is_tensor(embs):
                embs = torch.tensor(embs)

            embs = torch.nn.functional.normalize(embs, p=2, dim=1)

            return embs.detach().cpu()
        
        # Jina v4
        if "jina-embeddings-v3" in self.model_name:
            texts_to_encode = texts
            if not is_query and titles is not None and len(texts) == len(titles):
                print("Concatenating titles and texts for document encoding.")
                texts_to_encode = [f"{title} {text}" for title, text in zip(titles, texts)]

            task = "retrieval.query" if is_query else "retrieval.passage"
            embs = self.model.encode(texts_to_encode, task=task, show_progress_bar=True)
            
            embs = torch.stack([torch.from_numpy(t).float().to("cpu") for t in embs], dim=0)
            #embs = torch.nn.functional.normalize(embs, p=2, dim=1)

            return embs

        # concat titles for docs
        texts_to_encode = texts
        if not is_query and titles is not None:
            if len(texts) == len(titles):
                print("Concatenating titles and texts for document encoding.")
                if self._dual_family == "dpr":
                    sep = self.context_tokenizer.sep_token if hasattr(self, "context_tokenizer") else "[SEP]"
                    texts_to_encode = [f'{title} {sep} {text}' for title, text in zip(titles, texts)]
                else:
                    texts_to_encode = [f'{title} {text}' for title, text in zip(titles, texts)]
            else:
                print(f"Warning: Mismatch in length of titles ({len(titles)}) and texts ({len(texts)}). Ignoring titles.")

        all_embeddings = []

        # Default pooling
        pooling = self.config.get('pooling')
        if pooling is None:
            if self._handler_type in ('dual_encoder', 'peft_biencoder'):
                pooling = 'mean'
            else:
                pooling = 'mean'

        desc = f"Encoding ({'Queries' if is_query else 'Docs'}) | Pooling: {pooling}"
        for i in tqdm(range(0, len(texts_to_encode), batch_size), desc=desc):
            batch_texts = texts_to_encode[i:i + batch_size]

            # Choose model/tokenizer
            if self._handler_type == 'dual_encoder':
                if is_query:
                    model, tokenizer = self.question_model, self.question_tokenizer
                else:
                    model, tokenizer = self.context_model, self.context_tokenizer
            else:
                model, tokenizer = self.model, getattr(self, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError("Tokenizer is not initialized for this handler.")

            model.eval()

            # Apply instruction (for single/peft)
            if self._handler_type in ('transformer', 'peft_biencoder'):
                batch_texts = self._apply_instruction(batch_texts, is_query=is_query)

            if self._handler_type == "peft_biencoder":
                if "promptriever" in self.model_name:
                    raw = tokenizer(
                        batch_texts,
                        max_length=512 - 1, # for eos token
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        padding=False,
                        truncation=True,
                    )
                    eos_id = tokenizer.eos_token_id
                    raw["input_ids"] = [ids + [eos_id] for ids in raw["input_ids"]]
                    inputs = tokenizer.pad(
                        raw,
                        padding=True,
                        pad_to_multiple_of=8,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024) # 2048/4096 for best results for RepLLaMA
            else:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)

                    if self._handler_type == 'dual_encoder':
                        if "dpr" in self.model_name["question"]:
                            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                                embs = outputs.pooler_output
                            else:
                                embs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state[:, 0]
                        else:
                            embs = self._pooling(outputs, inputs['attention_mask'], pooling)
                    else:
                        if "promptriever" in self.model_name:
                            last_hidden = outputs.last_hidden_state
                            idxs = inputs['attention_mask'].sum(dim=1).long() - 1
                            embs = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idxs]
                        else:
                            embs = self._pooling(outputs, inputs['attention_mask'], pooling)

                    if self.config.get("normalize", True):
                        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

                    all_embeddings.append(embs.detach().cpu())

        return torch.cat(all_embeddings, dim=0)

    # ---------------------------------------------------------------
    # -------------------- RERANKING (TART-FULL) --------------------
    # ---------------------------------------------------------------
    @torch.inference_mode()
    def rerank(
        self,
        queries,
        docs_per_query,
        instruction: str = None,
        batch_size: int = 64,
        max_docs: int = 100
    ):
        """
        Reranks documents for queries using TART (EncT5).

        Following the official usage:
        - First sequence:  "{instruction} [SEP] {query}"
        - Second sequence: passage
        - Score: softmax(logits, dim=1)[positive_class]
        """
        if self._handler_type != 'tart_reranker':
            raise RuntimeError("rerank() is only available for TART-full instances.")

        # ----- Default instruction -----
        if instruction is None:
            instruction = "Find the passage that answers the given query."

        # Normalize input shapes
        if isinstance(queries, str):
            queries = [queries]

        # If docs_per_query is a flat list of strings, wrap once
        if isinstance(docs_per_query, list) and docs_per_query and isinstance(docs_per_query[0], str):
            docs_per_query = [docs_per_query]

        # ----- Positive class index -----
        pos_idx = 1
        if hasattr(self.model.config, "label2id") and isinstance(self.model.config.label2id, dict):
            l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
            for key in ["true", "yes", "relevant", "entailment", "1"]:
                if key in l2i:
                    pos_idx = l2i[key]
                    break

        # ----- Separator token -----
        # The official TART examples hard-code "[SEP]" in the string.
        sep_token_str = "[SEP]"

        all_query_scores = []

        for q, cand_docs in tqdm(
            zip(queries, docs_per_query),
            desc="TART Reranking",
            total=len(queries)
        ):
            # Limit number of candidates per query that we rerank
            cand_docs = cand_docs[:max_docs]

            if not cand_docs:
                all_query_scores.append([])
                continue

            # First sequence: instruction + SEP + query
            head = f"{instruction} {sep_token_str} {q}"
            heads = [head] * len(cand_docs)

            current_query_scores = []

            # Process in mini-batches
            for i in range(0, len(cand_docs), batch_size):
                batch_passages = cand_docs[i:i + batch_size]
                batch_heads    = heads[i:i + batch_size]

                # Two-sequence tokenization, as in the TART examples:
                # tokenizer(["instr [SEP] q", ...], [p1, p2, ...])
                inputs = self.tokenizer(
                    batch_heads,
                    batch_passages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True, # "longest_first"
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits  # [B, num_labels]

                # Convert logits -> probabilities and take positive class prob
                if logits.dim() == 2 and logits.size(1) > 1:
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    scores = probs[:, pos_idx].float().cpu().tolist()
                else:
                    # Fallback: single logit; just use it directly
                    scores = logits.squeeze(-1).float().cpu().tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                current_query_scores.extend(scores)

            all_query_scores.append(current_query_scores)

        return all_query_scores
