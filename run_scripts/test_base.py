#!/usr/bin/env python
# coding: utf-8
import os
import sys

def setup_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["JAVA_TOOL_OPTIONS"] = os.environ.get("JAVA_TOOL_OPTIONS", "") + \
        " -Djava.util.logging.config.file=/full/path/jul.properties"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["NCCL_DEBUG"] = "INFO"

    # Project root on sys.path (safe)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

def main():
    setup_env()

    import jnius_config
    jnius_config.set_options('-Xmx16G', '-Dorg.apache.lucene.search.BooleanQuery.maxClauseCount=8192')

    from transformers import set_seed
    from dotenv import load_dotenv
    import faiss
    import torch
    import pandas as pd
    import json

    from src.data_handler import DataHandler
    from src.evaluator import Evaluator
    from src.configs.datasets import DATASETS, SMALL_DATASETS
    from src.run import run
    from src.misc import create_folder_structure, get_dataset_variants

    NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"Using {NUM_THREADS} CPU threads based on SLURM_CPUS_PER_TASK.", flush=True)
    faiss.omp_set_num_threads(NUM_THREADS)
    torch.set_num_threads(NUM_THREADS)

    # Device
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} CUDA device(s)", flush=True)
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    set_seed(42)

    # Load env + HF auth ONLY in parent process
    load_dotenv()
    hf_key = os.getenv("HUGGINGFACE_KEY")

    # Avoid login() unless absolutely needed
    if hf_key:
        from huggingface_hub import login
        login(hf_key)  # runs only in parent now

    create_folder_structure(only_local=True)

    handler = DataHandler(
        sources=DATASETS,
        folder="data/raw"
    )
    handler.save()

    results_per_query = {}
    runs_cache = {}

    for model in ["bm25"]:
        for dataset in DATASETS: # ["irds:beir/cqadupstack/android", "irds:beir/cqadupstack/english", "irds:beir/cqadupstack/gaming", "irds:beir/cqadupstack/gis", "irds:beir/cqadupstack/mathematica", "irds:beir/cqadupstack/physics", "irds:beir/cqadupstack/programmers", "irds:beir/cqadupstack/stats", "irds:beir/cqadupstack/tex", "irds:beir/cqadupstack/unix", "irds:beir/cqadupstack/webmasters", "irds:beir/cqadupstack/wordpress"]:
            base_label = dataset.replace("/", "_").replace(":", "_")
            run_key = (dataset, model)

            for variant in get_dataset_variants(handler, dataset):
                print(f"\n▶ Running {model} on {dataset}; variant: {variant}", flush=True)
                out = run(model, handler, dataset, DEVICE, variant=variant, save_report=True, archive=False)

                bucket = runs_cache.setdefault(run_key, {})
                bucket[variant] = out

                if set(bucket.keys()) >= {"og", "changed"}:
                    evaluator = Evaluator(dataset, skip_self_matches="auto")

                    qrels_og = handler.read(dataset, variant="og")[2]
                    qrels_ch = handler.read(dataset, variant="changed")[2]
                    qrels_og_df = pd.DataFrame(list(qrels_og))
                    qrels_ch_df = pd.DataFrame(list(qrels_ch))

                    run_og = bucket["og"]["results"]
                    run_ch = bucket["changed"]["results"]

                    p_mrr_macro, p_mrr_perq = evaluator.p_mrr(qrels_og_df, qrels_ch_df, run_og, run_ch, k=None)
                    print(f"[{model} | {dataset}] p-MRR = {p_mrr_macro*100:.3f}", flush=True)

                    og_agg = bucket["og"]["metrics_agg"]
                    if "ndcg_cut_5" in og_agg:
                        std_name = "ndcg_cut_5"
                    elif "mean_avg_precision" in og_agg:
                        std_name = "mean_avg_precision"
                    else:
                        std_name = sorted(og_agg.keys())[0] if og_agg else "standard_metric"

                    std_value = float(og_agg.get(std_name, float("nan")))

                    elapsed_og = bucket["og"]["timing"]
                    elapsed_ch = bucket["changed"]["timing"]

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
                    print(f"💾 Saved combined report to {combined_path}", flush=True)

if __name__ == "__main__":
    main()

