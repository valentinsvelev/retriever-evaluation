################################################################################
# mappings.py
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

# ------------------------------------------------------
# --- Figure mappings ----------------------------------
# ------------------------------------------------------
MODEL_FAMILIES = {
    "sparse": ["bm25", "doc2query", "deepct", "sparta", "unicoil", "splade"],
    "dense": ["dpr", "colbert", "tasb", "ance", "simcse", "retromae", "cocondenser", "gtr", "contriever", "cocodr", "dragon", "hyde", "simlm", "query2doc"],
    "instruction_tuned": ["instructor", "llm2vec", "repllama"],
    "embedding_models": ["e5", "gte", "bge", "jina", "gritlm", "nomic", "qwen", "nvembed", "gemma", "kalm"]
}

MODEL_FAMILIES_PRETTY = {
    "Sparse Retrievers": ["bm25", "doc2query", "deepct", "sparta", "unicoil", "splade", "query2doc"],
    "Dense Retrievers": ["dpr", "colbert", "tasb", "ance", "simcse", "retromae", "cocondenser", "gtr", "contriever", "cocodr", "dragon", "hyde", "simlm"],
    "Instruction-Tuned Retrievers": ["instructor", "llm2vec", "repllama"],
    "General Embedding Models": ["e5", "gte", "bge", "jina", "gritlm", "nomic", "qwen", "nvembed", "gemma", "kalm"]
}

model_to_family = {
    model: family
    for family, models in MODEL_FAMILIES_PRETTY.items()
    for model in models
}

FAMILY_COLORS = {
    "Sparse Retrievers": "#1f77b4",
    "Dense Retrievers": "#ff7f0e",
    "Instruction-Tuned Retrievers": "#2ca02c",
    "General Embedding Models": "#9467bd"
}

FAMILY_MARKERS = {
    "Sparse Retrievers": "s",               # square (filled)
    "Dense Retrievers": "o",                # circle
    "Instruction-Tuned Retrievers": "^",    # triangle up
    "General Embedding Models": "D",        # diamond
    "Other": "v",                           # triangle down
}

MODEL_NAMES_PRETTY = {
    # Sparse
    "bm25": "BM25",
    "doc2query": "docT5query",
    "deepct": "DeepCT",
    "sparta": "SPARTA",
    "unicoil": "uniCOIL",
    "splade": "SPLADE-v3",
    "query2doc": "query2doc",

    # Dense
    "dpr": "DPR",
    "colbert": "ColBERTv2",
    "tasb": "TAS-B",
    "ance": "ANCE",
    "simcse": "SimCSE",
    "retromae": "RetroMAE",
    "cocondenser": "coCondenser",
    "gtr": "GTR (large)",
    "contriever": "Contriever",
    "cocodr": "COCO-DR (large)",
    "dragon": "DRAGON+",
    "simlm": "SimLM",
    "hyde": "HyDE",

    # Instruction-tuned
    "instructor": "InstructOR (large)",
    "llm2vec": "LLM2Vec",
    "repllama": "RepLLaMA",

    # General embedding models
    "e5": "mE5 (large)",
    "gte": "GTE (large v1.5)",
    "bge": "BGE (large v1.5)",
    "jina": "Jina Embeddings v3",
    "gritlm": "GritLM",
    "nomic": "Nomic Embed (v1.5)",
    "qwen": "Qwen3 Embedding (0.6B)",
    "nvembed": "NV-Embed-v2",
    "gemma": "EmbeddingGemma",
    "kalm": "KaLM-Embedding (v2.5)",
}

DATASET_TITLES_PRETTY = {
    "msmarco-passage": "MS MARCO",
    "msmarco-passage-trec-dl-2019": "TREC-DL 2019",
    "msmarco-passage-trec-dl-2020": "TREC-DL 2020",
    "beir": "BEIR",
    "lotte": "LoTTE",
    "instructir_robustness": "InstructIR",
    "followir_score": "FollowIR",
}

KEEP_LABELS = {
    "msmarco-passage": {
        "BM25": {"dx": -6, "dy": 10, "ha": "right", "va": "center"},
        "docT5query": {"dx": 10, "dy": 0, "ha": "left", "va": "center"},
        "DeepCT": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 12, "dy": -13},
        "uniCOIL": {"dx": 0, "dy": 15},
        "SPLADE-v3": {"dx": 25, "dy": 12},
        "DPR": {"dx": 0, "dy": 10},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": 25, "dy": 0},
        "query2doc": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "RepLLaMA": {"dx": -6, "dy": 0, "ha": "right"},
        "NV-Embed-v2": {"dx": 35, "dy": 10},
        "KaLM-Embedding (v2.5)": {"dx": 2, "dy": -12, "ha": "left"},
        "GritLM": {"dx": -6, "dy": -10, "ha": "right"},
    },

    "msmarco-passage-trec-dl-2019": {
        "BM25": {"dx": -6, "dy": 13, "ha": "right"},
        "docT5query": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 8, "dy": -8, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "DPR": {"dx": 15, "dy": 10},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": -15, "dy": -15},
        "GritLM": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "ANCE": {"dx": 0, "dy": -15},
        "KaLM-Embedding (v2.5)": {"dx": 6, "dy": -10, "ha": "left"},
    },

    "msmarco-passage-trec-dl-2020": {
        "BM25": {"dx": -6, "dy": 11, "ha": "right"},
        "docT5query": {"dx": -6, "dy": 15},
        "DeepCT": {"dx": 8, "dy": -8, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "DPR": {"dx": -15, "dy": 10},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": 15, "dy": -15},
        "InstructOR (large)": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "KaLM-Embedding (v2.5)": {"dx": -25, "dy": -15, "ha": "left"},
        "query2doc": {"dx": 3, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "ANCE": {"dx": -6, "dy": -13, "ha": "right"},
        "RepLLaMA": {"dx": -6, "dy": 6, "ha": "right"},
        "GritLM": {"dx": 6, "dy": -13, "ha": "left"},
        "LLM2Vec": {"dx": 0, "dy": -15}
    },

    "beir": {
        "SPLADE-v3": {"dx": 0, "dy": -15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "query2doc": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "GritLM": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 0, "dy": -12, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": 10, "ha": "left"},
        "DPR": {"dx": 0, "dy": -15},
        "SimCSE": {"dx": 6, "dy": -10, "ha": "left"},
        "GTE (large v1.5)": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": -15},
    },

    "lotte": {
        "BM25": {"dx": -6, "dy": -13, "ha": "right"},
        "DeepCT": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "query2doc": {"dx": 0, "dy": -15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "NV-Embed-v2": {"dx": -35, "dy": 8},
        "GritLM": {"dx": 18, "dy": 10},
        "RepLLaMA": {"dx": 0, "dy": -15},
        "KaLM-Embedding (v2.5)": {"dx": 6, "dy": -13, "ha": "left"},
        "uniCOIL": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 0, "dy": -15, "ha": "right"},
        "DPR": {"dx": 0, "dy": -15},
        "SimLM": {"dx": 15, "dy": -12},
        "ColBERTv2": {"dx": 12, "dy": 15, "ha": "right"},
    },

    "instructir_robustness": {
        "BM25": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 0, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "docT5query": {"dx": 15, "dy": -15},
        "DeepCT": {"dx": 0, "dy": 15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": -15, "dy": 15},
        "query2doc": {"dx": 0, "dy": 15},
        "GritLM": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "DPR": {"dx": 6, "dy": 10, "ha": "left"},
        "SimCSE": {"dx": 0, "dy": -15},
    },

    "followir_score": {
        "BM25": {"dx": 0, "dy": -15, "ha": "right"},
        "SPLADE-v3": {"dx": 0, "dy": 15, "ha": "left"},
        "uniCOIL": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 25, "dy": 0},
        "query2doc": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15, "ha": "left"},
        "RepLLaMA": {"dx": -6, "dy": -15, "ha": "left"},
        "LLM2Vec": {"dx": 6, "dy": 15},
        "GritLM": {"dx": 0, "dy": 15, "ha": "right"},
        "SimLM": {"dx": -9, "dy": 0, "ha": "right"},
        "docT5query": {"dx": 0, "dy": -15, "ha": "left"},
    },
}

KEEP_LABELS_PRESENTATION = {
    "msmarco-passage": {
        "BM25": {"dx": -6, "dy": -15},
        "docT5query": {"dx": 10, "dy": -7, "ha": "left", "va": "center"},
        "DeepCT": {"dx": 25, "dy": 0},
        "SPARTA": {"dx": 0, "dy": -15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "SPLADE-v3": {"dx": 25, "dy": 15},
        "DPR": {"dx": 0, "dy": 12},
        "ColBERTv2": {"dx": 0, "dy": 10, "ha": "right"},
        "SimCSE": {"dx": 0, "dy": -15},
        "query2doc": {"dx": 0, "dy": -15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "RepLLaMA": {"dx": -6, "dy": 0, "ha": "right"},
        "NV-Embed-v2": {"dx": 35, "dy": 10},
        "KaLM-Embedding (v2.5)": {"dx": 15, "dy": 0},
        "GritLM": {"dx": -6, "dy": -10, "ha": "right"},
    },
    
    "msmarco-passage-trec-dl-2019": {
        "BM25": {"dx": -1, "dy": 13, "ha": "right"},
        "docT5query": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 8, "dy": -8, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "DPR": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": 0, "dy": -15},
        "GritLM": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "ANCE": {"dx": 0, "dy": -15},
        "KaLM-Embedding (v2.5)": {"dx": 6, "dy": -10, "ha": "left"},
    },

    "msmarco-passage-trec-dl-2020": {
        "BM25": {"dx": -2, "dy": 11, "ha": "right"},
        "docT5query": {"dx": -6, "dy": 11},
        "DeepCT": {"dx": 2, "dy": -15, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "DPR": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "SimCSE": {"dx": 0, "dy": -15},
        "InstructOR (large)": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "KaLM-Embedding (v2.5)": {"dx": 10, "dy": -15},
        "query2doc": {"dx": 8, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "ANCE": {"dx": -6, "dy": -13, "ha": "right"},
        "RepLLaMA": {"dx": -6, "dy": 6, "ha": "right"},
        "GritLM": {"dx": 6, "dy": -13, "ha": "left"},
        "LLM2Vec": {"dx": 0, "dy": -15}
    },

    "beir": {
        "SPLADE-v3": {"dx": 0, "dy": -15},
        #"ColBERTv2": {"dx": 0, "dy": 15},
        "query2doc": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "GritLM": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 0, "dy": -15, "ha": "left"},
        "SPARTA": {"dx": 0, "dy": 12, "ha": "left"},
        "DPR": {"dx": 0, "dy": -15},
        "SimCSE": {"dx": 6, "dy": -10, "ha": "left"},
        "GTE (large v1.5)": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": -15},
    },
    
    "lotte": {
        "BM25": {"dx": -2, "dy": -13, "ha": "right"},
        "DeepCT": {"dx": 0, "dy": -15},
        "SimCSE": {"dx": 0, "dy": -15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "query2doc": {"dx": 0, "dy": -15},
        "LLM2Vec": {"dx": 0, "dy": -15},
        "NV-Embed-v2": {"dx": -35, "dy": 8},
        "GritLM": {"dx": 18, "dy": 10},
        "RepLLaMA": {"dx": 0, "dy": -15},
        "KaLM-Embedding (v2.5)": {"dx": 6, "dy": -13, "ha": "left"},
        "uniCOIL": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 0, "dy": 15, "ha": "right"},
        "DPR": {"dx": 0, "dy": -15},
        "SimLM": {"dx": 15, "dy": -12},
        "ColBERTv2": {"dx": -8, "dy": 20},
    },

    "instructir_robustness": {
        "BM25": {"dx": 0, "dy": -15},
        "SPARTA": {"dx": 0, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "docT5query": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 0, "dy": 15},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "query2doc": {"dx": 0, "dy": 15},
        "GritLM": {"dx": 0, "dy": 15},
        "LLM2Vec": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15},
        "RepLLaMA": {"dx": 0, "dy": 15},
        "DPR": {"dx": 6, "dy": 10, "ha": "left"},
        "SimCSE": {"dx": 0, "dy": -15},
    },
    
    "followir_score": {
        "BM25": {"dx": 0, "dy": -15, "ha": "right"},
        "SPLADE-v3": {"dx": 0, "dy": 15},
        "uniCOIL": {"dx": 0, "dy": 15},
        "SPARTA": {"dx": 0, "dy": -15},
        "query2doc": {"dx": 0, "dy": 15},
        "DeepCT": {"dx": 0, "dy": 15},
        "ColBERTv2": {"dx": 0, "dy": 15},
        "NV-Embed-v2": {"dx": 0, "dy": 15, "ha": "left"},
        "RepLLaMA": {"dx": -6, "dy": -15, "ha": "left"},
        "LLM2Vec": {"dx": 6, "dy": 15},
        "GritLM": {"dx": -12, "dy": 15},
        "SimLM": {"dx": -3, "dy": 10, "ha": "right"},
        "docT5query": {"dx": 0, "dy": -15, "ha": "left"},
    },
}

MODEL_TO_FAMILY = {
    model: family
    for family, models in MODEL_FAMILIES_PRETTY.items()
    for model in models
}

PRETTY_TO_FAMILY = {
    MODEL_NAMES_PRETTY.get(model, model): family
    for model, family in MODEL_TO_FAMILY.items()
}

# DATASET_SIZES = {
#     "irds:msmarco-passage/dev/small": {
#         "queries": 6980, "docs": 8841823,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },

#     # BEIR
#     "irds:beir/trec-covid": {
#         "queries": 50, "docs": 171332,
#         "avg_query_len": 10.60,
#         "avg_doc_len": 160.77,
#     },
#     "irds:beir/nfcorpus/test": {
#         "queries": 323, "docs": 3633,
#         "avg_query_len": 3.30,
#         "avg_doc_len": 232.26,
#     },
#     "irds:beir/nq": {
#         "queries": 3452, "docs": 2681468,
#         "avg_query_len": 9.16,
#         "avg_doc_len": 78.88,
#     },
#     "irds:beir/hotpotqa": {
#         "queries": 7405, "docs": 5233329,
#         "avg_query_len": 17.61,
#         "avg_doc_len": 46.30,
#     },
#     "irds:beir/fiqa/test": {
#         "queries": 648, "docs": 57638,
#         "avg_query_len": 10.77,
#         "avg_doc_len": 132.32,
#     },
#     "irds:beir/arguana": {
#         "queries": 1406, "docs": 8674,
#         "avg_query_len": 192.98,
#         "avg_doc_len": 166.80,
#     },
#     "irds:beir/webis-touche2020/v2": {
#         "queries": 49, "docs": 382545,
#         "avg_query_len": 6.55,
#         "avg_doc_len": 292.37,
#     },
#     "irds:beir/cqadupstack": {
#         "queries": 13145, "docs": 457199,
#         "avg_query_len": 8.59,
#         "avg_doc_len": 129.09,
#     },
#     "irds:beir/quora/test": {
#         "queries": 10000, "docs": 522931,
#         "avg_query_len": 9.53,
#         "avg_doc_len": 11.44,
#     },
#     "irds:beir/dbpedia-entity/test": {
#         "queries": 400, "docs": 4635922,
#         "avg_query_len": 5.39,
#         "avg_doc_len": 49.68,
#     },
#     "irds:beir/scidocs": {
#         "queries": 1000, "docs": 25657,
#         "avg_query_len": 9.38,
#         "avg_doc_len": 176.19,
#     },
#     "irds:beir/fever/test": {
#         "queries": 6666, "docs": 5416568,
#         "avg_query_len": 8.13,
#         "avg_doc_len": 84.76,
#     },
#     "irds:beir/climate-fever": {
#         "queries": 1535, "docs": 5416535,
#         "avg_query_len": 20.13,
#         "avg_doc_len": 84.76,
#     },
#     "irds:beir/scifact/test": {
#         "queries": 300, "docs": 5183,
#         "avg_query_len": 12.37,
#         "avg_doc_len": 213.63,
#     },

#     # LoTTE pooled
#     "irds:lotte/pooled/test/search": {
#         "queries": 3869, "docs": 2819103,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },
#     "irds:lotte/pooled/test/forum": {
#         "queries": 10025, "docs": 2819103,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },

#     # InstructIR
#     "hf:kaist-ai/InstructIR": {
#         "queries": 9906, "docs": 16072,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },

#     # FollowIR
#     "hf:jhu-clsp/news21-instructions": {
#         "queries": 32, "docs": 30900,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },
#     "hf:jhu-clsp/core17-instructions": {
#         "queries": 20, "docs": 19900,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },
#     "hf:jhu-clsp/robust04-instructions": {
#         "queries": 52, "docs": 47500,
#         "avg_query_len": None,
#         "avg_doc_len": None,
#     },
# }


# ------------------------------------------------------
# --- Table mappings -----------------------------------
# ------------------------------------------------------
COL_GROUPS = [
    ("MS MARCO",   [(r"MRR$@10$", "msmarco-passage")]),
    ("TREC-DL 19", [(r"nDCG$@10$", "msmarco-passage-trec-dl-2019")]),
    ("TREC-DL 20", [(r"nDCG$@10$", "msmarco-passage-trec-dl-2020")]),
    ("BEIR",       [(r"nDCG$@10$", "beir")]),
    ("LoTTE",      [(r"S$@5$", "lotte")]),
    ("InstructIR", [(r"R$@10$", "instructir_robustness"), (r"nDCG$@10$", "instructir_ndcg")]),
    ("FollowIR", [("Score", "followir_score"), (r"$p$-MRR", "followir_pmrr")]),
]

ROW_BLOCKS = [
    ("Sparse Retrievers", [
        (r"BM25 \citep{robertson1994okapi}", "bm25"),
        (r"docT5query \citep{nogueira2019doct5query}", "doc2query"),
        (r"query2doc \citep{wang2023query2doc}", "query2doc"),
    ]),
    ("Learned Sparse Retrievers", [
        (r"DeepCT \citep{dai2020deepct}", "deepct"),
        (r"SPARTA \citep{zhao2021sparta}", "sparta"),
        (r"uniCOIL \citep{lin2021unicoil}", "unicoil"),
        (r"SPLADE-v3 \citep{lassance2024spladev3}", "splade"),
    ]),
    ("Dense Retrievers", [
        (r"DPR \citep{karpukhin2020dpr}", "dpr"),
        (r"ColBERTv2 \citep{santhanam2022colbertv2}", "colbert"),
        (r"TAS-B \citep{hofstatter2021tasb}", "tasb"),
        (r"ANCE \citep{xiong2021ance}", "ance"),
        (r"SimCSE \citep{gao2021simcse}", "simcse"),
        (r"RetroMAE \citep{xiao2022retromae}", "retromae"),
        (r"coCondenser \citep{gao2022cocondenser}", "cocondenser"),
        (r"GTR (large) \citep{ni2022gtr}", "gtr"),
        (r"Contriever \citep{izacard2022contriever}", "contriever"),
        (r"COCO-DR (large) \citep{yu2022cocodr}", "cocodr"),
        (r"\textsc{SimLM} \citep{wang2023simlm}", "simlm"),
        (r"\textsc{Dragon}+ \citep{lin2023dragon}", "dragon"),
        (r"HyDE \citep{gao2023hyde}", "hyde"),
    ]),
    ("Instruction-Tuned Retrievers", [
        (r"\textsc{InstructOR} (large) \citep{su2023instructor}", "instructor"),
        (r"LLM2Vec \citep{behnamghader2024llmvec}", "llm2vec"),
        (r"RepLLaMA \citep{ma2024repllama}", "repllama"),
    ]),
    ("General Embedding Models", [
        (r"mE5 (large) \citep{wang2024me5}", "e5"),
        (r"GTE (large v1.5) \citep{li2023gte}", "gte"),
        (r"BGE (large v1.5) \citep{xiao2024bge}", "bge"),
        (r"Jina Embeddings v3 \citep{sturua2024jina3}", "jina"),
        (r"Nomic Embed (v1.5) \citep{nussbaum2025nomic}", "nomic"),
        (r"\textsc{GritLM} \citep{muennighoff2025gritlm}", "gritlm"),
        (r"Qwen3 Embedding (0.6B) \citep{zhang2025qwen3embedding}", "qwen"),
        (r"NV-Embed-v2 \citep{lee2025nvembed}", "nvembed"),
        (r"EmbeddingGemma \citep{vera2025embeddinggemma}", "gemma"),
        (r"KaLM-Embedding (v2.5) \citep{zhao2026kalmembeddingv2}", "kalm"),
    ]),
]

ROW_BLOCKS_NOCITE = [
    ("Sparse Retrievers", [
        (r"BM25", "bm25"),
        (r"docT5query", "doc2query"),
        (r"query2doc", "query2doc"),
    ]),
    ("Learned Sparse Retrievers", [
        (r"DeepCT", "deepct"),
        (r"SPARTA", "sparta"),
        (r"uniCOIL", "unicoil"),
        (r"SPLADE-v3", "splade"),
    ]),
    ("Dense Retrievers", [
        (r"DPR", "dpr"),
        (r"ColBERTv2", "colbert"),
        (r"TAS-B", "tasb"),
        (r"ANCE", "ance"),
        (r"SimCSE", "simcse"),
        (r"RetroMAE", "retromae"),
        (r"coCondenser", "cocondenser"),
        (r"GTR (large)", "gtr"),
        (r"Contriever", "contriever"),
        (r"COCO-DR (large)", "cocodr"),
        (r"\textsc{SimLM}", "simlm"),
        (r"\textsc{Dragon}+", "dragon"),
        (r"HyDE", "hyde"),
    ]),
    ("Instruction-Tuned Retrievers", [
        (r"\textsc{InstructOR} (large)", "instructor"),
        (r"LLM2Vec", "llm2vec"),
        (r"RepLLaMA", "repllama"),
    ]),
    ("General Embedding Models", [
        (r"mE5 (large)", "e5"),
        (r"GTE (large v1.5)", "gte"),
        (r"BGE (large v1.5)", "bge"),
        (r"Jina Embeddings v3", "jina"),
        (r"Nomic Embed (v1.5)", "nomic"),
        (r"\textsc{GritLM}", "gritlm"),
        (r"Qwen3 Embedding (0.6B)", "qwen"),
        (r"NV-Embed-v2", "nvembed"),
        (r"EmbeddingGemma", "gemma"),
        (r"KaLM-Embedding (v2.5)", "kalm"),
    ]),
]

BEIR_ORDER = [
    "irds:beir/trec-covid",
    "irds:beir/nfcorpus/test",
    "irds:beir/nq",
    "irds:beir/hotpotqa/test",
    "irds:beir/fiqa/test",
    "irds:beir/arguana",
    "irds:beir/webis-touche2020/v2",
    "irds:beir/cqadupstack",
    "irds:beir/quora/test",
    "irds:beir/dbpedia-entity/test",
    "irds:beir/scidocs",
    "irds:beir/fever/test",
    "irds:beir/climate-fever",
    "irds:beir/scifact/test",
]

BEIR_COL_NAMES = {
    "irds:beir/trec-covid": "trec-covid",
    "irds:beir/nfcorpus/test": "nfcorpus",
    "irds:beir/nq": "nq",
    "irds:beir/hotpotqa/test": "hot",
    "irds:beir/fiqa/test": "fiqa",
    "irds:beir/arguana": "arg",
    "irds:beir/webis-touche2020/v2": "touche",
    "irds:beir/cqadupstack": "cqa",
    "irds:beir/quora/test": "quora",
    "irds:beir/dbpedia-entity/test": "dbpedia",
    "irds:beir/scidocs": "scidocs",
    "irds:beir/fever/test": "fever",
    "irds:beir/climate-fever": "climate-fever",
    "irds:beir/scifact/test": "scifact",
}

MSMARCO_TREC_DATASETS = [
    "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged",
    "irds:msmarco-passage/trec-dl-2020/judged",
]
MSMARCO_ID = "irds:msmarco-passage/dev/small"
TREC19_ID = "irds:msmarco-passage/trec-dl-2019/judged"
TREC20_ID = "irds:msmarco-passage/trec-dl-2020/judged"

LOTTE_SEARCH_ID = "irds:lotte/pooled/test/search"
LOTTE_FORUM_ID  = "irds:lotte/pooled/test/forum"

INSTRUCTIR_ID = "hf:kaist-ai/InstructIR"
FOLLOWIR_ROBUST_ID = "hf:jhu-clsp/robust04-instructions"
FOLLOWIR_NEWS21_ID = "hf:jhu-clsp/news21-instructions"
FOLLOWIR_CORE17_ID = "hf:jhu-clsp/core17-instructions"

FOLLOWIR_ORDER = [
    FOLLOWIR_ROBUST_ID,
    FOLLOWIR_NEWS21_ID,
    FOLLOWIR_CORE17_ID,
]
