MODEL_FAMILIES = {
    "sparse": ["bm25", "doc2query", "deepct", "sparta", "unicoil", "splade"],
    "dense": ["dpr", "colbert", "tasb", "ance", "simcse", "retromae", "cocondenser", "gtr", "contriever", "cocodr", "dragon", "hyde", "simlm", "query2doc"],
    "instruction_tuned": ["tart", "instructor", "llm2vec", "repllama"],
    "embedding_models": ["e5", "gte", "bge", "jina", "gritlm", "nomic", "qwen", "nvembed", "gemma", "kalm", "sfr"]
}

MODEL_FAMILIES_PRETTY = {
    "Sparse Retrievers": ["bm25", "doc2query", "deepct", "sparta", "unicoil", "splade"],
    "Dense Retrievers": ["dpr", "colbert", "tasb", "ance", "simcse", "retromae", "cocondenser", "gtr", "contriever", "cocodr", "dragon", "hyde", "simlm", "query2doc"],
    "Instruction-Tuned Retrievers": ["tart", "instructor", "llm2vec", "repllama"],
    "General Embedding Models": ["e5", "gte", "bge", "jina", "gritlm", "nomic", "qwen", "nvembed", "gemma", "kalm", "sfr"]
}

model_to_family = {
    model: family
    for family, models in MODEL_FAMILIES_PRETTY.items()
    for model in models
}

MODEL_NAMES_PRETTY = {
    # Sparse
    "bm25": "BM25",
    "doc2query": "doc2query",
    "deepct": "DeepCT",
    "sparta": "SPARTA",
    "unicoil": "uniCOIL",
    "splade": "SPLADE",

    # Dense
    "dpr": "DPR",
    "colbert": "ColBERT",
    "tasb": "TAS-B",
    "ance": "ANCE",
    "simcse": "SimCSE",
    "retromae": "RetroMAE",
    "cocondenser": "CoCondenser",
    "gtr": "GTR",
    "contriever": "Contriever",
    "cocodr": "COCO-DR",
    "dragon": "DRAGON",
    "simlm": "SimLM",
    "hyde": "HyDE",
    "query2doc": "query2doc",

    # Instruction-tuned
    "tart": "TART",
    "instructor": "InstructOR",
    "llm2vec": "LLM2Vec",
    "repllama": "RepLLaMA",
    "promptriever": "Promptriever",

    # General embedding models
    "e5": "E5",
    "gte": "GTE",
    "bge": "BGE",
    "jina": "Jina Embeddings",
    "gritlm": "GritLM",
    "nomic": "Nomic Embed",
    "qwen": "Qwen3 Embedding",
    "nvembed": "NV-Embed",
    "gemma": "EmbeddingGemma",
    "kalm": "KaLM-Embedding",
    "sfr": "SFR-Embedding"
}

DATASET_SIZES = {
    "irds:msmarco-passage/dev/small":               {"queries": 6980,  "docs": 8841823},
    "irds:msmarco-passage/trec-dl-2019/judged":    {"queries": 43,    "docs": 8841823},
    "irds:msmarco-passage/trec-dl-2020/judged":    {"queries": 54,    "docs": 8841823},

    # BEIR
    "irds:beir/trec-covid":                         {"queries": 50,    "docs": 171332},
    "irds:beir/nfcorpus/test":                      {"queries": 323,   "docs": 3633},
    "irds:beir/nq":                                 {"queries": 3452,  "docs": 2681468},
    "irds:beir/hotpotqa":                           {"queries": 7405,  "docs": 5233329},
    "irds:beir/fiqa/test":                          {"queries": 648,   "docs": 57638},
    "irds:beir/arguana":                            {"queries": 1406,  "docs": 8674},
    "irds:beir/webis-touche2020/v2":                {"queries": 49,    "docs": 382545},
    "irds:beir/cqadupstack":                        {"queries": 13145, "docs": 457199},
    "irds:beir/quora/test":                         {"queries": 10000, "docs": 522931},
    "irds:beir/dbpedia-entity/test":                {"queries": 400,   "docs": 4635922},
    "irds:beir/scidocs":                            {"queries": 1000,  "docs": 25657},
    "irds:beir/fever/test":                         {"queries": 6666,  "docs": 5416568},
    "irds:beir/climate-fever":                      {"queries": 1535,  "docs": 5416535},
    "irds:beir/scifact/test":                       {"queries": 300,   "docs": 5183},

    # LoTTE pooled
    "irds:lotte/pooled/test/search":                {"queries": 3869,  "docs": 2819103},
    "irds:lotte/pooled/test/forum":                 {"queries": 10025, "docs": 2819103},

    # InstructIR
    "hf:kaist-ai/InstructIR":                       {"queries": 9906,  "docs": 16072},

    # FollowIR
    "hf:jhu-clsp/news21-instructions":              {"queries": 32,    "docs": 30900},
    "hf:jhu-clsp/core17-instructions":              {"queries": 20,    "docs": 19900},
    "hf:jhu-clsp/robust04-instructions":            {"queries": 52,    "docs": 47500},
}
