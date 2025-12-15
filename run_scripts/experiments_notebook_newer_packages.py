#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# In[ ]:


from transformers import set_seed
from huggingface_hub import login
from dotenv import load_dotenv
import faiss
import torch
import pandas as pd
import json

NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
print(f"Using {NUM_THREADS} CPU threads based on SLURM_CPUS_PER_TASK.")
faiss.omp_set_num_threads(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

from src.data_handler import DataHandler
from src.evaluator import Evaluator
from src.configs.datasets import DATASETS, LARGE_DATASETS, SMALL_DATASETS
from src.configs.models import generalist_embedders
from src.run import run
from src.misc import create_folder_structure, get_dataset_variants


# In[ ]:


try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def _compat_get_usable_length(self, seq_length, layer_idx=None):
            """
            transformers>=4.46 expects get_usable_length(seq_length, layer_idx).
            Older DynamicCache exposes get_seq_length([layer_idx]) or get_seq_length().
            We return the cached length (past tokens) for the layer if available,
            otherwise fall back to a global length or 0.
            """
            # Preferred: per-layer length if supported
            if hasattr(self, "get_seq_length"):
                try:
                    # Some versions accept layer_idx
                    return self.get_seq_length(layer_idx)
                except TypeError:
                    # Older versions: no args
                    return self.get_seq_length()
            # Last resort: no past available
            return 0

        DynamicCache.get_usable_length = _compat_get_usable_length
except Exception as e:
    print("[warn] DynamicCache patch failed:", e)


# In[ ]:


# Set device
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} CUDA device(s)")
    # Use generic "cuda" so DataParallel / device_map can work
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device: " + str(DEVICE))

# Set seed
set_seed(42)

# Get all API keys for paid models and HF models
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")
gemini_key = os.getenv("GOOGLEAI_KEY")
hf_key = os.getenv("HUGGINGFACE_KEY")

# Log into HF for locked models
login(hf_key)


# In[ ]:


create_folder_structure(only_local=True)


# In[ ]:


handler = DataHandler(
    sources=DATASETS,
    folder="data/raw"
)

handler.save()


# In[ ]:


results_per_query = {}
runs_cache = {} # {(dataset, model): {"og": {...}, "changed": {...}}}

if __name__ == "__main__":
    for model in ["gemma", "kalm", "gritlm", "instructor"]:
        for dataset in DATASETS:
            base_label = dataset.replace("/", "_").replace(":", "_")
            run_key = (dataset, model)
    
            for variant in get_dataset_variants(handler, dataset):
                print(f"\n▶ Running {model} on {dataset}; variant: {variant}")
                out = run(model, handler, dataset, DEVICE, variant=variant, save_report=True, archive=False)
    
                # cache everything we need later
                bucket = runs_cache.setdefault(run_key, {})
                bucket[variant] = out  # contains metrics_agg, metrics_perq, results, elapsed
    
                # once both present, compute p-MRR and write the single report
                if set(bucket.keys()) >= {"og", "changed"}:
                    evaluator = Evaluator(dataset, skip_self_matches="auto")
    
                    # qrels for each variant
                    qrels_og = handler.read(dataset, variant="og")[2]
                    qrels_ch = handler.read(dataset, variant="changed")[2]
                    qrels_og_df = pd.DataFrame(list(qrels_og))
                    qrels_ch_df = pd.DataFrame(list(qrels_ch))
    
                    # raw runs for p-MRR
                    run_og = bucket["og"]["results"]
                    run_ch = bucket["changed"]["results"]
    
                    # compute p-MRR
                    p_mrr_macro, p_mrr_perq = evaluator.p_mrr(qrels_og_df, qrels_ch_df, run_og, run_ch, k=None)
                    print(f"[{model} | {dataset}] p-MRR = {p_mrr_macro*100:.3f}")
    
                    # pull the standard metric from OG (MAP for robust/core17, nDCG@5 for news21)
                    og_agg = bucket["og"]["metrics_agg"]
                    # prefer ndcg_cut_5 if present (news21), else mean_avg_precision (robust/core17)
                    if "ndcg_cut_5" in og_agg:
                        std_name = "ndcg_cut_5"
                    elif "mean_avg_precision" in og_agg:
                        std_name = "mean_avg_precision"
                    else:
                        # fallback: take the first key deterministically
                        std_name = sorted(og_agg.keys())[0] if og_agg else "standard_metric"
    
                    std_value = float(og_agg.get(std_name, float("nan")))
    
                    # elapsed
                    elapsed_og = bucket["og"]["timing"]
                    elapsed_ch = bucket["changed"]["timing"]
    
                    # write the single combined report
                    out_dir = f"outputs/scores/{model}"
                    os.makedirs(out_dir, exist_ok=True)
                    combined_path = os.path.join(out_dir, f"{base_label}.json")
    
                    combined_report = {
                        "model_name": model,
                        "dataset_id": dataset,
                        "metrics": {std_name: std_value, "p_mrr": float(p_mrr_macro)},
                        "summary_stats": {
                            "og": bucket["og"]["summary_stats"],
                            "changed": bucket["changed"]["summary_stats"],
                        },
                        "runtime": {
                            "og": elapsed_og,
                            "changed": elapsed_ch,
                        }
                    }
    
                    with open(combined_path, "w", encoding="utf-8") as f:
                        json.dump(combined_report, f, indent=2)
                    print(f"💾 Saved combined report to {combined_path}")


# In[ ]:


# !PYPROJECT_NAME="master_thesis" PYPROJECT_VERSION="0.0.1" python -c 'import sys,subprocess,re,os; fr=subprocess.check_output([sys.executable,"-m","pip","freeze","--disable-pip-version-check"], text=True); ansi=re.compile(r"\x1b\[[0-9;]*[A-Za-z]"); deps=sorted([ln.strip() for ln in ansi.sub("",fr).splitlines() if ln.strip() and ln.split("==")[0].split("@")[0].split("[")[0].lower() not in {"pip","setuptools","wheel","pkg-resources","distribute"}], key=str.lower); name=os.environ.get("PYPROJECT_NAME","my-project"); ver=os.environ.get("PYPROJECT_VERSION","0.1.0"); rp=f">={sys.version_info.major}.{sys.version_info.minor}"; lines=["[build-system]","requires = [\"hatchling>=1.0.0\"]","build-backend = \"hatchling.build\"","", "[project]", f"name = \"{name}\"", f"version = \"{ver}\"", f"requires-python = \"{rp}\"", "dependencies = ["]; lines += [f"    \"{d}\"," for d in deps[:-1]] + ([f"    \"{deps[-1]}\"" ] if deps else []); lines.append("]"); open("pyproject.toml","w",encoding="utf-8").write("\n".join(lines)+"\n")'


# In[ ]:


# import sys, subprocess, pathlib
# out = pathlib.Path.cwd() / "requirements_new.txt"
# with open(out, "w") as f:
#     subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=f)

