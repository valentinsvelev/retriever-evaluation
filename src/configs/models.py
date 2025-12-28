sparse_models = {
    # -------------------------
    # --- Sparse retrievers ---
    # -------------------------
    "bm25": {},
    "doc2query": {"model_path": "doc2query/msmarco-t5-base-v1"},
    "deepct": {"model_path": "macavaney/deepct"},
    "unicoil": {"model_path": "castorini/unicoil-noexp-msmarco-passage"},
    "sparta": {"model_path": "BeIR/sparta-msmarco-distilbert-base-v1"},
    "splade": {"model_path": "naver/splade-v3"}, # naver/splade-cocondenser-ensembledistil
}

dense_models = {
    # ------------------------
    # --- Dense retrievers ---
    # ------------------------
    "dpr": {
        "model_path_ctx": "facebook/dpr-ctx_encoder-multiset-base", # "facebook/dpr-ctx_encoder-single-nq-base",
        "model_path_q": "facebook/dpr-question_encoder-multiset-base", # "facebook/dpr-question_encoder-single-nq-base",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "cocodr": {
        "model_path": "OpenMatch/cocodr-large-msmarco",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "tasb": {
        "model_path": "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco", # sentence-transformers/msmarco-distilbert-base-tas-b
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "dragon": {
        "model_path_ctx": "facebook/dragon-plus-context-encoder", # facebook/dragon-roberta-context-encoder
        "model_path_q": "facebook/dragon-plus-query-encoder", # facebook/dragon-roberta-query-encoder
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "contriever": {
        "model_path": "facebook/contriever-msmarco",
        "pooling": "mean",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "retromae": {
        "model_path": "Shitao/RetroMAE_BEIR", #"nthakur/RetroMAE_MSMARCO_finetune", #"Shitao/RetroMAE_MSMARCO_distill",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "cocondenser": {
        "model_path": "Luyu/co-condenser-marco-retriever",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": False
    },
    "ance": {
        "model_path": "castorini/ance-msmarco-passage", # sentence-transformers/msmarco-roberta-base-ance-firstp
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": True
    },
    "gtr": {
        "model_path": "sentence-transformers/gtr-t5-large",
    },
    "simcse": {
        "model_path": "princeton-nlp/unsup-simcse-roberta-large",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": True
    },
    "hyde": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct" # google/flan-t5-xxl
    },
    "query2doc": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct" # google/flan-t5-xxl
    },
    "simlm": {
        "model_path": "intfloat/simlm-base-msmarco-finetuned",
        "pooling": "cls",
        "query_instruction": None,
        "doc_instruction": None,
        "normalize": True
    },
}

instruction_tuned_retrievers = {
    # -----------------------------------
    # --- Intruction-tuned retrievers ---
    # -----------------------------------
    "tart": {
        "model_path": "facebook/tart-full-flan-t5-xl" # facebook/tart-full-t0-3b
    },
    "llm2vec": {
        "model_path": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
        "pooling": "last_token",
        "query_instruction": "Given a web search query, retrieve relevant passages that answer the query:",
        "doc_instruction": None,
        "normalize": True
    },
    "instructor": {
        "model_path": "hkunlp/instructor-large",
        "query_instruction": "Represent the question: ",
        "doc_instruction": "Represent the document: ",
        "pooling": "mean",
        "normalize": True
    },
    # "promptriever": {
    #     "model_path": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    #     "pooling": "last_token",
    #     "query_instruction": "query:  ",
    #     "doc_instruction": "passage:  ",
    #     "normalize": True
    # },
    "repllama": {
        "model_path": "castorini/repllama-v1-7b-lora-passage",
        "pooling": "last_token",
        "query_instruction": "query: {}</s>",
        "doc_instruction": "passage: {}</s>",
        "normalize": True
    },
}

generalist_embedders = {
    # --------------------------------
    # --- General embedding models ---
    # --------------------------------
    "bge": {
        "model_path": "BAAI/bge-large-en-v1.5",
        "normalize": True
    },
    "e5": {
        "model_path": "intfloat/multilingual-e5-large-instruct", # intfloat/e5-mistral-7b-instruct, intfloat/e5-large-v2
        "pooling": "mean",
        "query_instruction": "Instruct: Retrieve passages that answer this question.\nQuery: ", # None,
        "doc_instruction": "Passage: ", # None,
        "normalize": True
    },
    "gte": {
        "model_path": "Alibaba-NLP/gte-large-en-v1.5", # Alibaba-NLP/gte-Qwen2-7B-instruct, Alibaba-NLP/gte-multilingual-base
        "pooling": "cls", # "last_token"
        "query_instruction": None, # "query: ",
        "doc_instruction": None, # "passage: "
    },
    # "arctic": {
    #     "model_path": "Snowflake/snowflake-arctic-embed-l-v2.0",
    #     "pooling": "cls",
    #     "query_instruction": "query: ",
    #     "doc_instruction": ""
    # },
    "nomic": {
        "model_path": "nomic-ai/nomic-embed-text-v1.5",
        "pooling": "mean",
        "query_instruction": "search_query: ",
        "doc_instruction": "search_document: "
    },
    "jina": {
        "model_path": "jinaai/jina-embeddings-v3",
    },
    "qwen": {
        "model_path": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "last_token",
        "query_instruction": "Instruct: Retrieve passages that answer this query\nQuery: ",
        "doc_instruction": None,
        "normalize": True
    },
    "gritlm": {
        "model_path": "GritLM/GritLM-7B",
        "pooling": "mean",
        "query_instruction": "Given a question, retrieve passages that answer the question:",
        "doc_instruction": ""
    },
    "gemma": {
        "model_path": "google/embeddinggemma-300m",
        "pooling": "mean",
        "query_instruction": "",
        "doc_instruction": ""
    },
    "nvembed": {
        "model_path": "nvidia/NV-Embed-v2",
        "query_instruction": "Given a question, retrieve passages that answer the question",
        "doc_instruction": None
    },
    "kalm": {
        "model_path": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    },
    # "sfr": {
    #     "model_path": "Salesforce/SFR-Embedding-Mistral",
    #     "pooling": "last_token",
    #     "query_instruction": "Given a question, retrieve passages that answer the question",
    #     "doc_instruction": None,
    #     "normalize": True
    # }
}

MODELS = {**sparse_models, **dense_models, **instruction_tuned_retrievers, **generalist_embedders}

del instruction_tuned_retrievers["llm2vec"]
del generalist_embedders["nvembed"]
