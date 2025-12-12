import os
import json
import gzip
import pandas as pd
from typing import List
from pathlib import Path
from collections import defaultdict

from src.data_handler import DataHandler
from src.evaluator import Evaluator
from src.analysis.mappings import DATASET_SIZES, MODEL_FAMILIES_PRETTY, MODEL_NAMES_PRETTY
from src.analysis.misc import bootstrap_mean_stats, assign_family, prettify_model_name, label_to_dataset_id_for_sizes


# --------------------------------------------------
# --- MAPPINGS -------------------------------------
# --------------------------------------------------

# --- BEIR -----------------------------------------
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

# --- MS MARCO + TREC DL ---------------------------
MSMARCO_TREC_DATASETS = [
    "irds:msmarco-passage/dev/small",
    "irds:msmarco-passage/trec-dl-2019/judged",
    "irds:msmarco-passage/trec-dl-2020/judged",
]
MSMARCO_ID = "irds:msmarco-passage/dev/small"
TREC19_ID = "irds:msmarco-passage/trec-dl-2019/judged"
TREC20_ID = "irds:msmarco-passage/trec-dl-2020/judged"

# --- LoTTE pooled ---------------------------------
LOTTE_SEARCH_ID = "irds:lotte/pooled/test/search"
LOTTE_FORUM_ID  = "irds:lotte/pooled/test/forum"

# --- InstructIR + FollowIR ------------------------
INSTRUCTIR_ID = "hf:kaist-ai/InstructIR"
FOLLOWIR_ROBUST_ID = "hf:jhu-clsp/robust04-instructions"
FOLLOWIR_NEWS21_ID = "hf:jhu-clsp/news21-instructions"
FOLLOWIR_CORE17_ID = "hf:jhu-clsp/core17-instructions"

FOLLOWIR_ORDER = [
    FOLLOWIR_ROBUST_ID,
    FOLLOWIR_NEWS21_ID,
    FOLLOWIR_CORE17_ID,
]

# --------------------------------------------------
# --- CLASS ----------------------------------------
# --------------------------------------------------
class Aggregator():
    def __init__(self, scores_root: str, results_root: str, handler: DataHandler):
        self.scores_root = scores_root
        self.results_root = results_root
        self.handler = handler
    

    def _compute_perq_for_dataset(self, model_key: str, dataset_id: str, handler: DataHandler):
        """
        Recompute metrics_agg, metrics_perq, summary_stats using saved 'results' + qrels, with fallback filename logic:
        1) outputs/results/<model>/<dataset>.json
        2) outputs/results/<model>-<dataset>.json
        """
        ARCHIVE_ROOT = "/dataHDD1/masterthesis"
        dataset_label = dataset_id.replace("/", "_").replace(":", "_")

        # 1) Try the correct hierarchical path
        path1 = f"{ARCHIVE_ROOT}/outputs/results/{model_key}/{dataset_label}.json"

        # 2) Try the flattened filename fallback
        path2 = f"{ARCHIVE_ROOT}/outputs/results/{model_key}-{dataset_label}.json"

        # Determine which exists
        if os.path.exists(path1):
            results_path = path1
        elif os.path.exists(path2):
            results_path = path2
        else:
            raise FileNotFoundError(
                f"Could not find results for model '{model_key}' / dataset '{dataset_id}'.\n"
                f"Tried:\n - {path1}\n - {path2}"
            )

        # Load results
        try:
            with gzip.open(results_path, "rt", encoding="utf-8") as f:
                results = json.load(f)
        except:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        # Load qrels
        _, _, qrels_iter = handler.read(dataset_id, variant=None)
        qrels_df = pd.DataFrame(list(qrels_iter))

        # Evaluate
        evaluator = Evaluator(dataset_id, skip_self_matches="auto")
        metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels_df, results, print_msg=False)

        return metrics_agg, metrics_perq, summary_stats


    def _get_metric_from_scores(self, scores: dict, *candidates: str):
        """
        Return the first metric found in scores among the given candidates.
        """
        if not isinstance(scores, dict):
            return None
        for key in candidates:
            if key in scores:
                return scores[key]
        return None


    def _pivot_metric_any(self, df: pd.DataFrame, dataset_list: List[str], *metric_keys: str) -> pd.DataFrame:
        """
        Create a pivot table model x dataset for the *first* metric_key that exists in `scores`.
        This lets us be robust to slightly different metric naming (e.g., 'p_mrr' vs 'condensed_reciprocal_rank').
        """
        df_filtered = df[df["dataset_id"].isin(dataset_list)].copy()

        def _extract(scores):
            return self._get_metric_from_scores(scores, *metric_keys)

        metric_col = "_tmp_metric_" + "_".join(metric_keys)
        df_filtered[metric_col] = df_filtered["scores"].apply(_extract)

        pivot = df_filtered.pivot_table(
            index="model",
            columns="dataset_id",
            values=metric_col,
            aggfunc="first",
        )
        pivot = pivot.reindex(columns=dataset_list)
        return pivot


    def _group_models_by_family(self, model_index) -> list[tuple[str, list[str]]]:
        """
        Given an index/list of model names, return
        [(group_name, [models...]), ...] according to MODEL_FAMILIES_PRETTY.
        """
        used_models = set()
        grouped = []

        for group_name, patterns in MODEL_FAMILIES_PRETTY.items():
            group_models = []
            for pat in patterns:
                for model_raw in model_index:
                    if pat in model_raw.lower() and model_raw not in used_models:
                        group_models.append(model_raw)
                        used_models.add(model_raw)
            if group_models:
                grouped.append((group_name, group_models))

        return grouped
    
    def _generate_msmarco_trec_table(self, df: pd.DataFrame) -> str:
        datasets = MSMARCO_TREC_DATASETS

        pivot_mrr   = self._pivot_metric_any(df, datasets, "recip_rank_10")
        pivot_R1000 = self._pivot_metric_any(df, datasets, "recall_1000")
        pivot_ndcg  = self._pivot_metric_any(df, [TREC19_ID, TREC20_ID], "ndcg_cut_10")

        all_models = sorted(
            set(pivot_mrr.index)
            | set(pivot_R1000.index)
            | set(pivot_ndcg.index)
        )
        grouped = self._group_models_by_family(all_models)

        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")
        latex_lines.append(r"\caption{MS MARCO and TREC DL (2019 \& 2020) Results}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcccccc}")
        latex_lines.append(r"\toprule\toprule")
        latex_lines.append(
            r"\textbf{Dataset $\rightarrow$} & "
            r"\multicolumn{2}{c}{\textbf{MS MARCO}} & "
            r"\multicolumn{2}{c}{\textbf{TREC DL 2019}} & "
            r"\multicolumn{2}{c}{\textbf{TREC DL 2020}} \\"
        )
        latex_lines.append(
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
        )
        latex_lines.append(
            r"\textbf{Model $\downarrow$} & {\textbf{MRR}@10} & {\textbf{R}@1000} & "
            r"{\textbf{nDCG}@10} & {\textbf{R}@1000} & {\textbf{nDCG}@10} & {\textbf{R}@1000} \\"
        )
        latex_lines.append(r"\midrule")

        def safe_get(pivot, model, ds):
            if model not in pivot.index:
                return None
            v = pivot.loc[model].get(ds)
            return None if pd.isna(v) else v

        for group_name, group_models in grouped:
            latex_lines.append(r"\multicolumn{7}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)

                mrr_msm  = safe_get(pivot_mrr,   model_raw, MSMARCO_ID)
                R_msm    = safe_get(pivot_R1000, model_raw, MSMARCO_ID)

                ndcg_19  = safe_get(pivot_ndcg,  model_raw, TREC19_ID)
                R_19     = safe_get(pivot_R1000, model_raw, TREC19_ID)

                ndcg_20  = safe_get(pivot_ndcg,  model_raw, TREC20_ID)
                R_20     = safe_get(pivot_R1000, model_raw, TREC20_ID)

                vals = []
                for v in [mrr_msm, R_msm, ndcg_19, R_19, ndcg_20, R_20]:
                    if v is None:
                        vals.append(r"{}")
                    else:
                        vals.append(f"{v:.4f}")

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab:msmarco-trec-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def _generate_beir_table(self, df: pd.DataFrame) -> str:
        order = BEIR_ORDER
        col_names = BEIR_COL_NAMES

        # pivot for ndcg_cut_10 using our flexible helper
        pivot = self._pivot_metric_any(df, order, "ndcg_cut_10")

        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\scriptsize")
        latex_lines.append(r"\caption{BEIR Results (nDCG$@10$)}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{l" + "c"*len(order) + "}")
        latex_lines.append(r"\toprule\toprule")

        col_headers = " & ".join(col_names[d] for d in order)
        latex_lines.append(r"\textbf{Dataset $\rightarrow$} & " + col_headers + r" \\")
        latex_lines.append(r"\textbf{Model $\downarrow$} & " + " & ".join([""]*len(order)) + r" \\")
        latex_lines.append(r"\midrule")

        grouped = self._group_models_by_family(pivot.index)

        for group_name, group_models in grouped:
            latex_lines.append(
                r"\multicolumn{" + str(len(order)+1) + r"}{l}{\textbf{" + group_name + r"}}\\"
            )
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                row = pivot.loc[model_raw]
                pretty_name = prettify_model_name(model_raw)

                vals = []
                for d in order:
                    v = row.get(d)
                    if pd.isna(v):
                        vals.append(r"{}")
                    else:
                        vals.append(f"{v:.4f}")
                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab:beir-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def _generate_lotte_table(self, df: pd.DataFrame) -> str:
        datasets = [LOTTE_SEARCH_ID, LOTTE_FORUM_ID]
        pivot_s5 = self._pivot_metric_any(df, datasets, "success_5")

        grouped = self._group_models_by_family(pivot_s5.index)

        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")
        latex_lines.append(r"\caption{LoTTE Results (Success$@5$)}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcc}")
        latex_lines.append(r"\toprule\toprule")
        latex_lines.append(
            r"\textbf{Dataset $\rightarrow$} & \multicolumn{2}{c}{\textbf{Pooled}} \\"
        )
        latex_lines.append(
            r"\textbf{Model $\downarrow$} & {\textbf{Search}} & {\textbf{Forum}} \\"
        )
        latex_lines.append(r"\midrule")

        def safe_get(model, ds):
            if model not in pivot_s5.index:
                return None
            v = pivot_s5.loc[model].get(ds)
            return None if pd.isna(v) else v

        for group_name, group_models in grouped:
            latex_lines.append(r"\multicolumn{3}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)

                s_search = safe_get(model_raw, LOTTE_SEARCH_ID)
                s_forum  = safe_get(model_raw, LOTTE_FORUM_ID)

                vals = []
                for v in [s_search, s_forum]:
                    if v is None:
                        vals.append(r"{}")
                    else:
                        vals.append(f"{v:.4f}")

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab:lotte-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def _generate_instructir_followir_table(self, df: pd.DataFrame) -> str:
        # pivots for InstructIR MS MARCO
        pivot_instr_ndcg   = self._pivot_metric_any(df, [INSTRUCTIR_ID], "ndcg_cut_10")
        pivot_instr_robust = self._pivot_metric_any(df, [INSTRUCTIR_ID], "robustness_10")

        # FollowIR MAP / nDCG@5 / p-MRR
        pivot_map    = self._pivot_metric_any(df, [FOLLOWIR_ROBUST_ID, FOLLOWIR_CORE17_ID], "mean_avg_precision")
        pivot_ndcg5  = self._pivot_metric_any(df, [FOLLOWIR_NEWS21_ID], "ndcg_cut_5")
        pivot_p_mrr  = self._pivot_metric_any(df, FOLLOWIR_ORDER, "p_mrr")

        all_models = sorted(
            set(pivot_instr_ndcg.index)
            | set(pivot_instr_robust.index)
            | set(pivot_map.index)
            | set(pivot_ndcg5.index)
            | set(pivot_p_mrr.index)
        )
        grouped = self._group_models_by_family(all_models)

        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\scriptsize")
        latex_lines.append(r"\caption{\textsc{InstructIR} and FollowIR Results}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcccccccc}")
        latex_lines.append(r"\toprule\toprule")
        latex_lines.append(
            r"\textbf{Dataset $\rightarrow$} & "
            r"\multicolumn{2}{c}{\textbf{\textsc{InstructIR}}} & "
            r"\multicolumn{6}{c}{\textbf{FollowIR}} \\"
        )
        latex_lines.append(
            r"& \multicolumn{2}{c}{\textbf{MS MARCO}} "
            r"& \multicolumn{2}{c}{\textbf{Robust04}} "
            r"& \multicolumn{2}{c}{\textbf{News21}} "
            r"& \multicolumn{2}{c}{\textbf{Core17}} \\"
        )
        latex_lines.append(
            r"\textbf{Model $\downarrow$} & \textbf{nDCG}@10 & \textbf{Robustness}@10 & "
            r"\textbf{MAP} & $p$-\textbf{MRR} & \textbf{nDCG}@5 & $p$-\textbf{MRR} & "
            r"\textbf{MAP} & $p$-\textbf{MRR} \\"
        )
        latex_lines.append(r"\midrule")

        def safe_get(pivot, model, ds):
            if model not in pivot.index:
                return None
            v = pivot.loc[model].get(ds)
            return None if pd.isna(v) else v

        for group_name, group_models in grouped:
            latex_lines.append(r"\multicolumn{9}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)

                instr_ndcg   = safe_get(pivot_instr_ndcg,   model_raw, INSTRUCTIR_ID)
                instr_robust = safe_get(pivot_instr_robust, model_raw, INSTRUCTIR_ID)

                rob_map      = safe_get(pivot_map,   model_raw, FOLLOWIR_ROBUST_ID)
                rob_p_mrr    = safe_get(pivot_p_mrr, model_raw, FOLLOWIR_ROBUST_ID)

                news_ndcg5   = safe_get(pivot_ndcg5, model_raw, FOLLOWIR_NEWS21_ID)
                news_p_mrr   = safe_get(pivot_p_mrr, model_raw, FOLLOWIR_NEWS21_ID)

                core_map     = safe_get(pivot_map,   model_raw, FOLLOWIR_CORE17_ID)
                core_p_mrr   = safe_get(pivot_p_mrr, model_raw, FOLLOWIR_CORE17_ID)

                vals = []
                for v in [
                    instr_ndcg, instr_robust,
                    rob_map, rob_p_mrr,
                    news_ndcg5, news_p_mrr,
                    core_map, core_p_mrr,
                ]:
                    if v is None:
                        vals.append(r"{}")
                    else:
                        vals.append(f"{v:.4f}")

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab:instructir-followir-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def all_scores_to_df(self, aggregate_cqa: bool = True, add_metadata: bool = True) -> pd.DataFrame:
        rows = []

        for model_dir in self.scores_root.iterdir():
            if not model_dir.is_dir():
                continue

            for json_path in model_dir.glob("*.json"):

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                rows.append(
                    {
                        "model": model_dir.name,
                        "dataset": json_path.stem,
                        "scores": data.get("metrics"),
                        "summary_stats": data.get("summary_stats"),
                        "runtime": data.get("timing") or data.get("runtime"),
                    }
                )

        if aggregate_cqa:
            print("Aggregating CQADupstack datasets (averaging metrics, recomputing summary stats, summing up runtimes)...")
            # group cqadupstack rows by model
            cqa_by_model: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                if "cqadupstack" in str(r["dataset"]):
                    cqa_by_model[r["model"]].append(r)

            aggregated_rows: list[dict] = []

            for model, model_rows in cqa_by_model.items():
                # collect per-query metrics across all cqadupstack datasets
                perq_all: dict[str, dict[str, float]] = {}
                runtime_agg: dict[str, float] = defaultdict(float)

                for r in model_rows:
                    dataset_label = r.get("dataset")

                    if not dataset_label:
                        # if dataset_id missing, we can't recompute; skip this dataset
                        continue
                    
                    if dataset_label.startswith("irds_beir_cqadupstack_"):
                        suffix = dataset_label.split("irds_beir_cqadupstack_", 1)[1]
                        dataset_id = f"irds:beir/cqadupstack/{suffix}"
                    else:
                        # not a cqadupstack dataset we know how to map -> skip
                        continue

                    try:
                        metrics_agg, metrics_perq, summary_stats = self._compute_perq_for_dataset(
                            model_key=model,
                            dataset_id=dataset_id,
                            handler=self.handler,
                        )
                    except FileNotFoundError:
                        # silently skip datasets whose underlying results we don't have archived
                        continue

                    # make qids globally unique by prefixing dataset_id
                    for qid, mvals in metrics_perq.items():
                        full_qid = f"{dataset_id}::{qid}"
                        perq_all[full_qid] = mvals

                if not perq_all:
                    # nothing usable for this model's cqadupstack; skip
                    continue

                perq_df = pd.DataFrame.from_dict(perq_all, orient="index")

                # aggregated scores: mean over all queries
                agg_scores = perq_df.mean(axis=0).to_dict()

                # recompute summary stats via bootstrap on all queries
                agg_summary_stats: dict[str, dict] = {}
                for metric_name in perq_df.columns:
                    agg_summary_stats[metric_name] = bootstrap_mean_stats(perq_df[metric_name])

                # aggregate runtimes: sum all numeric timing fields
                for r in model_rows:
                    rt = r.get("runtime") or {}
                    for k, v in rt.items():
                        if isinstance(v, (int, float)):
                            runtime_agg[k] += v

                aggregated_rows.append(
                    {
                        "model": model,
                        "dataset": "irds_beir_cqadupstack",
                        "scores": agg_scores,
                        "summary_stats": agg_summary_stats,
                        "runtime": dict(runtime_agg),
                    }
                )

            # append aggregated rows to the original list
            rows.extend(aggregated_rows)

        df = pd.DataFrame(rows)
        
        if add_metadata:
            df["dataset_id"] = df["dataset"].apply(label_to_dataset_id_for_sizes)
            df["num_queries"] = df["dataset_id"].map(lambda d: DATASET_SIZES.get(d, {}).get("queries"))
            df["num_docs"] = df["dataset_id"].map(lambda d: DATASET_SIZES.get(d, {}).get("docs"))
            df['runtime_seconds'] = df['runtime'].apply(lambda x: x.get('runtime_seconds') if isinstance(x, dict) else None)
            df["family"] = df["model"].apply(assign_family)
            df["model_pretty"] = df["model"].apply(prettify_model_name)

        df_no_cqa = df[~df["dataset"].str.startswith("irds_beir_cqadupstack_")].copy()
        
        return df.sort_values(["model", "dataset"]), df_no_cqa.sort_values(["model", "dataset"])

    
    def generate_latex_table(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        dataset_name ∈ {"msmarco", "beir", "lotte", "instruct"}.
        For "instructir" and "followir" we generate the combined InstructIR+FollowIR table.
        """
        # store for helpers that inspect self.dataset_name, if any

        if dataset_name == "msmarco":
            return self._generate_msmarco_trec_table(df)
        elif dataset_name == "beir":
            return self._generate_beir_table(df)
        elif dataset_name == "lotte":
            return self._generate_lotte_table(df)
        elif dataset_name == "instruct":
            # both share the same combined table
            return self._generate_instructir_followir_table(df)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")