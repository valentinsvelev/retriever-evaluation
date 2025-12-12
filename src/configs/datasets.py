import ir_datasets
from collections import Counter


_SPLITS = {"train", "dev", "test"}

def _beir_base(ds: str) -> str:
    # keep everything up to (but not including) a final split token
    parts = ds.split("/")
    return "/".join(parts[:-1]) if parts[-1] in _SPLITS else ds

# All BEIR entries
_beir = [ds for ds in ir_datasets.registry if ds.startswith("beir/")]

# Count how many variants each base has
_beir_counts = Counter(_beir_base(ds) for ds in _beir)

# Keep singletons; for multi-variant bases keep only the /test split
filtered_beir = [
    ds for ds in _beir
    if _beir_counts[_beir_base(ds)] == 1 or ds.endswith("/test")
]

combined_ir_datasets = [
    ds for ds in ir_datasets.registry
    if ds in ["msmarco-passage/dev/small", "msmarco-passage/trec-dl-2019/judged", "msmarco-passage/trec-dl-2020/judged"] or ds.startswith("lotte/pooled/test/")
]

instruct_datasets = [
    "kaist-ai/InstructIR",
    "jhu-clsp/robust04-instructions",
    "jhu-clsp/core17-instructions",
    "jhu-clsp/news21-instructions",
    #"xlangai/BRIGHT"
]

filtered_beir = ["irds:" + w for w in filtered_beir]
combined_ir_datasets = ["irds:" + w for w in combined_ir_datasets]
instruct_datasets = ["hf:" + w for w in instruct_datasets]

DATASETS = filtered_beir + combined_ir_datasets + instruct_datasets

DATASETS.remove("irds:beir/msmarco/test")
DATASETS.remove("irds:beir/webis-touche2020")

LARGE_DATASETS = [
    "irds:beir/nq",
    "irds:lotte/pooled/test/search",
    "irds:lotte/pooled/test/forum",
    "irds:beir/dbpedia-entity/test",
    "irds:beir/hotpotqa/test",
    "irds:beir/climate-fever",
    "irds:beir/fever/test",
    "irds:msmarco-passage/dev/small"
]

SMALL_DATASETS = [x for x in DATASETS if x not in LARGE_DATASETS + ["irds:msmarco-passage/trec-dl-2019/judged", "irds:msmarco-passage/trec-dl-2020/judged"]]
