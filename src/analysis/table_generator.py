################################################################################
# table_generator.py
#
# Description: Code for creating latex tables using the results.
#
# Author: Valentin Velev
# Last updated: 14.02.2026
################################################################################

import os
import sys
import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from src.configs.datasets import filtered_beir
from src.data_handler import DataHandler
from src.evaluator import Evaluator
from src.analysis.mappings import (
    COL_GROUPS, ROW_BLOCKS, BEIR_ORDER, BEIR_COL_NAMES, 
    MSMARCO_TREC_DATASETS, MSMARCO_ID, TREC19_ID, TREC20_ID,
    LOTTE_FORUM_ID, LOTTE_SEARCH_ID, INSTRUCTIR_ID, 
    FOLLOWIR_CORE17_ID, FOLLOWIR_NEWS21_ID, FOLLOWIR_ROBUST_ID,
    FOLLOWIR_ORDER, MODEL_FAMILIES_PRETTY, ROW_BLOCKS_NOCITE
)
from src.configs.datasets import DATASETS
from src.analysis.misc import prettify_model_name

ROOT = Path.cwd().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TableGenerator:
    
    """
    Create an aggregate latex table and latex tables for each dataset.
    """
    
    def __init__(self, scores_root: str, data_handler: DataHandler):
        self.scores_root = scores_root
        self.data_handler = data_handler


    def _load_json(self, model_name: str, filename: str) -> dict:
        path = self.scores_root / model_name / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _get_metric(self, model_name: str, filename: str, metric: str) -> float:
        return float(self._load_json(model_name, filename)["metrics"][metric])


    def _get_first_metric(self, model_name: str, filename: str, *candidates: str) -> float:
        metrics = self._load_json(model_name, filename)["metrics"]
        for k in candidates:
            if k in metrics:
                return float(metrics[k])
        raise KeyError(f"None of {candidates} found in metrics for {model_name}/{filename}")


    def _dataset_id_to_label(self, dataset_id: str) -> str:
        return dataset_id.replace(":", "_").replace("/", "_")


    def _maybe_bold(self, val: float | None, max_val: float | None, digit: int) -> str:
        """
        Formats a value. Wraps in \textbf{} if it matches the column maximum.
        """
        if val is None:
            return r"{}"
        
        s = f"{val:.{digit}f}"
        
        # Check if max exists and value is effectively equal to it (handling float precision)
        if max_val is not None and abs(val - max_val) < 10 ** (-digit - 1):
            return rf"\textbf{{{s}}}"
        
        return s


    def _average_score(self, model_name: str, dataset_name: str, digit: int = 4, beir_avg: str = None) -> float:
        # ----------------
        # MS MARCO / TREC
        # ----------------
        MSMARCO_CASES = {
            "msmarco-passage": ("irds_msmarco-passage_dev_small.json", "recip_rank_10"),
            "msmarco-passage-trec-dl-2019": ("irds_msmarco-passage_trec-dl-2019_judged.json", "ndcg_cut_10"),
            "msmarco-passage-trec-dl-2020": ("irds_msmarco-passage_trec-dl-2020_judged.json", "ndcg_cut_10"),
        }
        if dataset_name in MSMARCO_CASES:
            fn, metric = MSMARCO_CASES[dataset_name]
            return round(self._get_metric(model_name, fn, metric), digit)

        # -----
        # BEIR
        # -----
        if dataset_name == "beir":
            scores = []
            cqa_scores = []

            for ds in filtered_beir:
                ds_norm = ds.replace("/", "_").replace(":", "_")
                v = self._get_metric(model_name, f"{ds_norm}.json", "ndcg_cut_10")
                if "cqadupstack" in ds_norm:
                    cqa_scores.append(v)
                else:
                    scores.append(v)

            cqa = sum(cqa_scores) / len(cqa_scores)

            if beir_avg == "macro":
                beir_val = (sum(scores) + cqa) / (len(scores) + 1)
            elif beir_avg == "micro":
                beir_val = (sum(scores) + sum(cqa_scores)) / (len(scores) + len(cqa_scores))

            return round(beir_val, digit)

        # -----
        # LoTTE
        # -----
        if dataset_name == "lotte":
            s_forum = self._get_metric(model_name, "irds_lotte_pooled_test_forum.json", "success_5")
            s_search = self._get_metric(model_name, "irds_lotte_pooled_test_search.json", "success_5")
            return round((s_forum + s_search) / 2, digit)

        # ----------
        # InstructIR
        # ----------
        if dataset_name in {"instructir_robustness", "instructir_ndcg"}:
            fn = "hf_kaist-ai_InstructIR.json"
            metric = "robustness_10" if dataset_name == "instructir_robustness" else "ndcg_cut_10"
            return round(self._get_metric(model_name, fn, metric), digit)

        # --------
        # FollowIR
        # --------
        if dataset_name == "followir_score":
            core_fn = "hf_jhu-clsp_core17-instructions_changed.json"
            news_fn = "hf_jhu-clsp_news21-instructions_changed.json"
            rob_fn  = "hf_jhu-clsp_robust04-instructions_changed.json"

            score_1 = self._get_first_metric(model_name, core_fn, "mean_avg_precision", "mean_avg_precision_1000") * 100
            score_2 = self._get_metric(model_name, news_fn, "ndcg_cut_5") * 100
            score_3 = self._get_first_metric(model_name, rob_fn,  "mean_avg_precision", "mean_avg_precision_1000") * 100

            return round((score_1 + score_2 + score_3) / 3, digit)

        if dataset_name == "followir_pmrr":
            files = [
                "hf_jhu-clsp_core17-instructions.json",
                "hf_jhu-clsp_news21-instructions.json",
                "hf_jhu-clsp_robust04-instructions.json",
            ]
            vals = [self._get_metric(model_name, fn, "p_mrr") * 100 for fn in files]
            return round(sum(vals) / 3, digit)

        raise KeyError(f"Unknown dataset_name: {dataset_name}")


    def _load_result_and_qrels(self, file_path: str, dataset_id: str):
        # Load results
        try:
            with gzip.open(f"{ROOT}/{file_path}.json", "rt", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            try:
                with gzip.open(f"{ROOT}/{file_path}.json.gz", "rt", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                print(f"{file_path} not found.")
                results = None

        # Variant handling:
        # - For FollowIR, the suffix is part of dataset_id on disk; DO NOT also use variant.
        # - For others, keep variant=None (or your existing logic if you truly need it).
        if "jhu-clsp" in dataset_id:
            variant = None
            dataset_id_for_read = dataset_id  # must include _og or _changed
        else:
            variant = None
            dataset_id_for_read = dataset_id

        _, _, qrels_iter = self.data_handler.read(dataset_id_for_read, variant=variant)
        qrels_df = pd.DataFrame(list(qrels_iter))

        # optional but recommended normalization
        if "qid" in qrels_df.columns:
            qrels_df["qid"] = qrels_df["qid"].astype(str)
        if isinstance(results, dict):
            results = {str(q): docs for q, docs in results.items()}

        return results, qrels_df


    def _file_path(self, model_name: str, dataset_label: str) -> str:
        if dataset_label.endswith("_og") or dataset_label.endswith("_changed"):
            return f"outputs/results/{model_name}/{dataset_label}"
        if "jhu-clsp" in dataset_label:
            return f"outputs/results/{model_name}/{dataset_label}_changed"
        return f"outputs/results/{model_name}/{dataset_label}"


    def _components_and_agg(self, model_name: str, ds_key: str, beir_avg: str):
        # ----------------
        # MS MARCO / TREC
        # ----------------
        if ds_key == "msmarco-passage":
            dataset_label = "irds_msmarco-passage_dev_small"
            dataset_id    = "irds:msmarco-passage/dev/small"
            metric_name   = "recip_rank_10"

            fp = self._file_path(model_name, dataset_label)
            results, qrels = self._load_result_and_qrels(fp, dataset_id)

            comps = [{
                "dataset_id": dataset_id,
                "results": results,
                "qrels": qrels,
                "metric_name": metric_name,
            }]
            return comps, (lambda vals: vals[0])

        if ds_key == "msmarco-passage-trec-dl-2019":
            dataset_label = "irds_msmarco-passage_trec-dl-2019_judged"
            dataset_id    = "irds:msmarco-passage/trec-dl-2019/judged"
            metric_name   = "ndcg_cut_10"

            fp = self._file_path(model_name, dataset_label)
            results, qrels = self._load_result_and_qrels(fp, dataset_id)

            comps = [{"dataset_id": dataset_id, "results": results, "qrels": qrels, "metric_name": metric_name}]
            return comps, (lambda vals: vals[0])

        if ds_key == "msmarco-passage-trec-dl-2020":
            dataset_label = "irds_msmarco-passage_trec-dl-2020_judged"
            dataset_id    = "irds:msmarco-passage/trec-dl-2020/judged"
            metric_name   = "ndcg_cut_10"

            fp = self._file_path(model_name, dataset_label)
            results, qrels = self._load_result_and_qrels(fp, dataset_id)

            comps = [{"dataset_id": dataset_id, "results": results, "qrels": qrels, "metric_name": metric_name}]
            return comps, (lambda vals: vals[0])

        # -----
        # BEIR
        # -----
        if ds_key == "beir":
            comps = []
            is_cqa = []

            for ds in filtered_beir:
                if ds.startswith("irds:") or ds.startswith("hf:"):
                    dataset_id = ds
                elif ds.startswith("beir/"):
                    dataset_id = "irds:" + ds
                else:
                    dataset_id = "irds:" + ds

                dataset_label = self._dataset_id_to_label(dataset_id)
                fp = self._file_path(model_name, dataset_label)

                results, qrels = self._load_result_and_qrels(fp, dataset_id)
                comps.append({
                    "dataset_id": dataset_id,
                    "results": results,
                    "qrels": qrels,
                    "metric_name": "ndcg_cut_10",
                })
                is_cqa.append("cqadupstack" in dataset_label)

            def agg(vals: list[float]) -> float:
                scores = [v for v, cqa in zip(vals, is_cqa) if not cqa]
                cqa_scores = [v for v, cqa in zip(vals, is_cqa) if cqa]
                cqa = sum(cqa_scores) / len(cqa_scores) if cqa_scores else 0.0
                denom_extra = 1 if cqa_scores else 0

                if beir_avg == "macro":
                    return (sum(scores) + cqa) / (len(scores) + denom_extra)
                elif beir_avg == "micro":
                    return (sum(scores) + sum(cqa_scores)) / (len(scores) + len(cqa_scores))
                else:
                    raise ValueError("beir_avg must be 'macro' or 'micro'")

            return comps, agg

        # -----
        # LoTTE
        # -----
        if ds_key == "lotte":
            forum_label = "irds_lotte_pooled_test_forum"
            forum_id    = "irds:lotte/pooled/test/forum"
            search_label = "irds_lotte_pooled_test_search"
            search_id    = "irds:lotte/pooled/test/search"

            comps = []
            for dataset_label, dataset_id in [(forum_label, forum_id), (search_label, search_id)]:
                fp = self._file_path(model_name, dataset_label)
                results, qrels = self._load_result_and_qrels(fp, dataset_id)
                comps.append({
                    "dataset_id": dataset_id,
                    "results": results,
                    "qrels": qrels,
                    "metric_name": "success_5",
                })

            return comps, (lambda vals: (vals[0] + vals[1]) / 2.0)

        # ----------
        # InstructIR
        # ----------
        if ds_key in {"instructir_robustness", "instructir_ndcg"}:
            dataset_label = "hf_kaist-ai_InstructIR"
            dataset_id    = "hf:kaist-ai/InstructIR"
            metric_name   = "robustness_10" if ds_key == "instructir_robustness" else "ndcg_cut_10"

            fp = self._file_path(model_name, dataset_label)
            results, qrels = self._load_result_and_qrels(fp, dataset_id)

            comps = [{"dataset_id": dataset_id, "results": results, "qrels": qrels, "metric_name": metric_name}]
            return comps, (lambda vals: vals[0])

        # --------
        # FollowIR
        # --------
        if ds_key == "followir_score":
            # These dataset_id strings must match your data_handler ids
            core_label, core_id, core_metric = "hf_jhu-clsp_core17-instructions", "hf:jhu-clsp/core17-instructions_changed", "mean_avg_precision"
            news_label, news_id, news_metric = "hf_jhu-clsp_news21-instructions", "hf:jhu-clsp/news21-instructions_changed", "ndcg_cut_5"
            rob_label,  rob_id,  rob_metric  = "hf_jhu-clsp_robust04-instructions","hf:jhu-clsp/robust04-instructions_changed","mean_avg_precision"

            comps = []
            for dataset_label, dataset_id, metric_name in [
                (core_label, core_id, core_metric),
                (news_label, news_id, news_metric),
                (rob_label,  rob_id,  rob_metric),
            ]:
                fp = self._file_path(model_name, dataset_label)
                results, qrels = self._load_result_and_qrels(fp, dataset_id)
                comps.append({"dataset_id": dataset_id, "results": results, "qrels": qrels, "metric_name": metric_name})

            def agg(vals: list[float]) -> float:
                # match your point estimate scaling
                return ((vals[0] * 100) + (vals[1] * 100) + (vals[2] * 100)) / 3.0

            return comps, agg

        if ds_key == "followir_pmrr":
            core_label, core_id = "hf_jhu-clsp_core17-instructions", "hf:jhu-clsp/core17-instructions"
            news_label, news_id = "hf_jhu-clsp_news21-instructions", "hf:jhu-clsp/news21-instructions"
            rob_label,  rob_id  = "hf_jhu-clsp_robust04-instructions","hf:jhu-clsp/robust04-instructions"

            comps = []
            for dataset_label, dataset_id in [(core_label, core_id), (news_label, news_id), (rob_label, rob_id)]:
                fp = self._file_path(model_name, dataset_label)
                results, qrels = self._load_result_and_qrels(fp, dataset_id)
                comps.append({"dataset_id": dataset_id, "results": results, "qrels": qrels, "metric_name": "p_mrr"})

            def agg(vals: list[float]) -> float:
                vals = [v * 100 for v in vals]
                return sum(vals) / 3.0

            return comps, agg

        raise KeyError(f"Unknown ds_key: {ds_key}")


    def _se_followir_pmrr(self, model_name: str, B: int = 300, seed: int = 0) -> float | None:
        pairs = [
            ("hf:jhu-clsp/core17-instructions_og",   "hf:jhu-clsp/core17-instructions_changed"),
            ("hf:jhu-clsp/news21-instructions_og",   "hf:jhu-clsp/news21-instructions_changed"),
            ("hf:jhu-clsp/robust04-instructions_og", "hf:jhu-clsp/robust04-instructions_changed"),
        ]

        evaluator = Evaluator("followir", skip_self_matches="auto")

        per_dataset_perq = []
        for og_id, ch_id in pairs:
            og_label = self._dataset_id_to_label(og_id)
            ch_label = self._dataset_id_to_label(ch_id)
            fp_og = f"outputs/results/{model_name}/{og_label}"
            fp_ch = f"outputs/results/{model_name}/{ch_label}"

            run_og, qrels_og = self._load_result_and_qrels(fp_og, og_id)
            run_ch, qrels_ch = self._load_result_and_qrels(fp_ch, ch_id)
            if run_og is None or run_ch is None or qrels_og is None or qrels_ch is None:
                return None

            macro, perq = evaluator.p_mrr(qrels_og, qrels_ch, run_og, run_ch, k=None)
            if perq is None or len(perq) == 0:
                return None

            vals = np.asarray(list(perq.values()), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None

            per_dataset_perq.append(vals)

        # Bootstrap the *aggregate* (mean of 3 datasets, then *100)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(B):
            ds_macros = []
            for vals in per_dataset_perq:
                n = len(vals)
                idx = rng.integers(0, n, size=n)
                ds_macros.append(float(vals[idx].mean()))
            samples.append((sum(ds_macros) / 3.0) * 100.0)

        return float(np.std(samples, ddof=1))


    def _perq_vector_from_evaluate(self, dataset_id: str, qrels: pd.DataFrame, run: dict, metric_name: str) -> np.ndarray | None:
        evaluator = Evaluator(dataset_id, skip_self_matches="auto")
        _, metrics_perq, _, robustness_per_group = evaluator.evaluate(qrels, run, print_msg=False)
        
        # # DEBUG
        # if "kaist-ai/InstructIR" in dataset_id:
        #     # print keys of first query's metrics
        #     first = next(iter(metrics_perq.values()), {})
        #     print("[InstructIR perq keys]", sorted(first.keys()))
        
        if metric_name == "robustness_10":
            if robustness_per_group is None:
                return None
            v = np.asarray(robustness_per_group, dtype=float)
            v = v[np.isfinite(v)]
            return None if v.size == 0 else v

        vals = []
        for qid, md in metrics_perq.items():
            if metric_name in md and md[metric_name] is not None and np.isfinite(md[metric_name]):
                vals.append(float(md[metric_name]))

        return None if len(vals) == 0 else np.asarray(vals, dtype=float)


    def _bootstrap_se_mean(self, vals: np.ndarray, B: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        n = vals.shape[0]
        idx = rng.integers(0, n, size=(B, n))
        means = vals[idx].mean(axis=1)
        return float(means.std(ddof=1))


    def _se_for_cell(self, model_name: str, ds_key: str, beir_avg: str, B: int = 1000, seed: int = 42) -> float | None:
        if ds_key == "followir_pmrr":
            return self._se_followir_pmrr(model_name, B=B, seed=seed)

        comps, agg = self._components_and_agg(model_name, ds_key, beir_avg)

        perq_vecs = []
        for c in comps:
            if c["results"] is None or c["qrels"] is None or len(c["qrels"]) == 0:
                return None
            v = self._perq_vector_from_evaluate(c["dataset_id"], c["qrels"], c["results"], c["metric_name"])
            if v is None:
                return None
            perq_vecs.append(v)

        # Bootstrap the aggregate: for each replicate, resample each component’s per-query vector
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(B):
            vals = []
            for v in perq_vecs:
                n = v.shape[0]
                idx = rng.integers(0, n, size=n)
                vals.append(float(v[idx].mean()))
            samples.append(float(agg(vals)))

        return float(np.std(samples, ddof=1))


    def create_aggregated_table(self, digit, beir_avg, published, boot: bool = False, boot_cutoffs: int | list | None = None, gen_std_dev_table: bool = False):
        """
        ...
        """

        avg_cache: dict[tuple[str, str], float | None] = {}
        se_cache: dict[tuple[str, str], float | None] = {}
        z_cache: dict[tuple[str, str], float | None] = {}

        def fmt(v):
            return r"{}" if v is None else f"{v:.{digit}f}"

        # def safe_avg(model: str, ds_key: str) -> float | None:
        #     k = (model, ds_key)
        #     if k in avg_cache:
        #         return avg_cache[k]
        #     try:
        #         v = self._average_score(model, ds_key, digit=digit, beir_avg=beir_avg)
        #     except Exception:
        #         v = None
        #     avg_cache[k] = v
        #     return v
        
        def safe_avg(model: str, ds_key: str) -> float | None:
            k = (model, ds_key)
            if k in avg_cache:
                return avg_cache[k]
            try:
                v = self._average_score(model, ds_key, digit=digit, beir_avg=beir_avg)
            except Exception as e:
                if ds_key == "instructir_robustness":
                    print(f"[safe_avg] model={model} ds_key={ds_key} ERROR: {type(e).__name__}: {e}")
                v = None
            avg_cache[k] = v
            return v
        
        def normalize_cutoffs(cutoffs):
            if cutoffs is None:
                return [2.0]
            if isinstance(cutoffs, (int, float)):
                return [float(cutoffs)]
            # assume iterable
            return sorted(float(c) for c in cutoffs)
        
        z_cutoffs = normalize_cutoffs(boot_cutoffs)

        def colorize_by_deviation(our: float | None, pub: float | None, ds: str, digit: int = 3) -> str:
            """
            Colors OUR value by (our - pub):
            blue  if abs(diff) <= 0.02
            green if diff > 0.02
            red   if diff < -0.02
            """
            if our is None:
                return r"{}"

            s = f"{our:.{digit}f}"
            thresh = 2 if ds in {"followir_score", "followir_pmrr"} else 0.02
            if pub is None or pd.isna(pub):
                return s

            diff = our - pub
            if abs(diff) <= thresh:
                return rf"\textcolor{{Blue}}{{{s}}}"
            elif diff > thresh:
                return rf"\textcolor{{ForestGreen}}{{{s}}}"
            else:
                return rf"\textcolor{{Red}}{{{s}}}"
        
        def colorize_by_z(z: float, s: str, cutoffs: list[float]) -> str:
            """
            cutoffs must be sorted positive values, e.g. [2] or [2, 3]
            """
            c0 = cutoffs[0]

            # central (blue) band
            if -c0 <= z <= c0:
                return rf"\textcolor{{SpecBlue}}{{{s}}}"

            # optional intermediate band (e.g. between 2 and 3)
            if len(cutoffs) > 1:
                c1 = cutoffs[1]

                if c0 < z <= c1:
                    return rf"\textcolor{{SpecTeal}}{{{s}}}"
                
                if z > c1:
                    return rf"\textcolor{{SpecGreen}}{{{s}}}"
                
                if -c1 <= z < -c0:
                    return rf"\textcolor{{SpecPurple}}{{{s}}}"
                
                if z < -c1:
                    return rf"\textcolor{{SpecRed}}{{{s}}}"

        def se_for_cell_cached(model: str, ds_key: str) -> float | None:
            k = (model, ds_key)
            if k in se_cache:
                return se_cache[k]
            se = self._se_for_cell(model, ds_key, beir_avg, B=1000, seed=42)
            if se is None or (not np.isfinite(se)) or se <= 0:
                se = None
            se_cache[k] = se
            return se

        def z_for_cell_cached(model: str, ds_key: str, our: float | None, pub: float | None) -> float | None:
            k = (model, ds_key)
            if k in z_cache:
                return z_cache[k]
            if our is None or pub is None or pd.isna(pub):
                z_cache[k] = None
                return None
            se = se_for_cell_cached(model, ds_key)
            if se is None:
                z_cache[k] = None
                return None
            z = (our - float(pub)) / se
            z_cache[k] = z
            return z
        
        def colorize_by_z_boot(model_name: str, our: float | None, pub: float | None, ds_key: str) -> str:
            if our is None:
                return r"{}"
            s = f"{our:.{digit}f}"

            # missing published → black
            if pub is None or pd.isna(pub):
                return s

            z = z_for_cell_cached(model_name, ds_key, our, pub)
            if z is None:
                return s

            return colorize_by_z(z, s, z_cutoffs)

        def z_cell_string(model_name: str, our: float | None, pub: float | None, ds_key: str) -> str:
            z = z_for_cell_cached(model_name, ds_key, our, pub)
            if z is None:
                return r"{}"
            return f"{z:.{digit}f}"

        def maybe_bold(val: float | None, ds_key: str, s: str) -> str:
            """
            Wrap LaTeX string s in \\textbf{} if val is the column maximum.
            """
            if val is None:
                return s
            max_v = col_max.get(ds_key)
            if max_v is not None and abs(val - max_v) < 10 ** (-digit):
                return rf"\textbf{{{s}}}"
            return s

        # flatten columns
        flat_cols = []
        for _, cols in COL_GROUPS:
            for metric_name, ds_key in cols:
                flat_cols.append((metric_name, ds_key))

        # find maxima (only for the metric table, as before)
        col_max = {}
        for _, ds_key in flat_cols:
            vals = []
            for _, rows in ROW_BLOCKS:
                for _, model_name in rows:
                    v = safe_avg(model_name, ds_key)
                    if v is not None:
                        vals.append(v)
            col_max[ds_key] = max(vals) if vals else None

        n_cols = 1 + len(flat_cols)

        def build_table(*, mode: str) -> str:
            """
            mode = "metric" or "z"
            """
            if boot:
                caption_colors = (
                    r"\textcolor{SpecBlue}{blue} = within 2 standard errors of published, "
                    r"\textcolor{SpecTeal}{teal} = between 2 and 3 standard errors above published, "
                    r"\textcolor{ForestGreen}{green} = 3 standard errors above published, "
                    r"\textcolor{SpecPurple}{purple} = between 2 and 3 standard errors below published, "
                    r"\textcolor{SpecRed}{red} = 3 standard errors below published."
                )
            else:
                caption_colors = (
                    r"\textcolor{Blue}{blue} = within 0.02 of published, "
                    r"\textcolor{ForestGreen}{green} = 0.02 higher than published, "
                    r"\textcolor{Red}{red} = 0.02 lower than published."
                )

            lines = []
            lines.append(r"\begin{table}[h!]")
            lines.append(r"    \centering")
            lines.append(r"    \scriptsize")

            if mode == "metric":
                caption = (
                    r"    \caption{Evaluation results -- all retrievers on all datasets. "
                    r"S$@5$ = Success$@5$ and R$@10$ = Robustness$@10$. "
                    r"Score is the average across MAP$@1000$ and nDCG$@5$. "
                    r"\textcolor{Black}{Black} = new result, "
                    + caption_colors +
                    r" The best result per dataset is highlighted bold.}"
                )
                label = r"    \label{tab:retrieval-quality}"
            else:
                caption = (
                    r"    \caption{Evaluation results -- bootstrapped z-scores. "
                    r"S$@5$ = Success$@5$ and R$@10$ = Robustness$@10$.}"
                )
                label = r"    \label{tab-a:retrieval-quality-zscores}"

            lines.append(caption)
            lines.append(rf"    \begin{{tabular}}{{l{'c'*len(flat_cols)}}}")
            lines.append(r"        \toprule")

            # Header row 1: datasets
            header_1 = r"        \textbf{Dataset} $\rightarrow$"
            for ds_name, cols in COL_GROUPS:
                if len(cols) == 1:
                    header_1 += rf" & \textbf{{{ds_name}}}"
                else:
                    header_1 += rf" & \multicolumn{{{len(cols)}}}{{c}}{{\textbf{{{ds_name}}}}}"
            header_1 += r" \\"
            lines.append(header_1)

            # cmidrules (your current hard-coded + group spans)
            col_cursor = 2
            cmid = []
            cmid.append(r"\cmidrule(lr){2-2}")
            cmid.append(r"\cmidrule(lr){3-3}")
            cmid.append(r"\cmidrule(lr){4-4}")
            cmid.append(r"\cmidrule(lr){5-5}")
            cmid.append(r"\cmidrule(lr){6-6}")

            for _, cols in COL_GROUPS:
                span = len(cols)
                if span > 1:
                    cmid.append(rf"\cmidrule(lr){{{col_cursor}-{col_cursor+span-1}}}")
                col_cursor += span
            if cmid:
                lines.append("        " + " ".join(cmid))

            # Header row 2: metrics
            header_2 = r"        \textbf{Model} $\downarrow$"
            for _, cols in COL_GROUPS:
                for metric_name, _ in cols:
                    header_2 += rf" & \textbf{{{metric_name}}}"
            header_2 += r" \\"
            lines.append(header_2)

            lines.append(r"        \midrule")

            # Body
            if mode == "z":
                blocks = ROW_BLOCKS_NOCITE
            else:
                blocks = ROW_BLOCKS
            
            for block_idx, (block_name, rows) in enumerate(blocks):
                lines.append(rf"        \multicolumn{{{n_cols}}}{{l}}{{\textbf{{{block_name}}}}}\\")
                lines.append(r"        \midrule")

                for disp_name, model_name in rows:
                    cells = []
                    for _, ds_key in flat_cols:
                        our = safe_avg(model_name, ds_key)
                        pub = published.loc[model_name, ds_key]

                        if mode == "metric":
                            if boot:
                                cell = colorize_by_z_boot(model_name, our, pub, ds_key)
                            else:
                                cell = colorize_by_deviation(our, pub, ds_key, digit)
                            cell = maybe_bold(our, ds_key, cell)
                        else:
                            # z-score table only makes sense in boot mode; otherwise we leave blanks
                            if boot:
                                cell = z_cell_string(model_name, our, pub, ds_key)
                            else:
                                cell = r"{}"

                        cells.append(cell)

                    lines.append(
                        "        " + disp_name + " & " + " & ".join(cells) + r" \\"
                    )

                if block_idx != len(blocks) - 1:
                    lines.append(r"        \midrule")

            lines.append(r"        \bottomrule")
            lines.append(r"    \end{tabular}")
            lines.append(label)
            lines.append(r"\end{table}")

            return "\n".join(lines)

        metric_table = build_table(mode="metric")

        # Your requested behavior:
        if boot and gen_std_dev_table:
            z_table = build_table(mode="z")
            return metric_table, z_table

        return metric_table


    def _get_score_for_disaggregated_table(self, model_name: str, dataset_name: str, metric: str):
        result = self._load_json(
            model_name, f"{self._dataset_id_to_label(dataset_name)}.json"
        )
        md = result["metrics"]

        if metric == "mean_avg_precision":
            return md.get("mean_avg_precision", md.get("mean_avg_precision_1000"))

        return md[metric]


    def create_msmarco_trec_table(self, digit: int = 3) -> str:
        # 1. PRE-CALCULATION PASS
        # Store scores in a dict: cache[model_name][key] = float
        cache = {}
        max_scores = {
            "msm_mrr": -1.0, "msm_r": -1.0,
            "dl19_n": -1.0,  "dl19_r": -1.0,
            "dl20_n": -1.0,  "dl20_r": -1.0
        }

        # Flatten models to iterate once for calculation
        all_models = [m for group in MODEL_FAMILIES_PRETTY.values() for m in group]

        for model_raw in all_models:
            scores = {}
            # MSMARCO
            scores["msm_mrr"] = self._get_score_for_disaggregated_table(model_raw, MSMARCO_ID, "recip_rank_10")
            scores["msm_r"]   = self._get_score_for_disaggregated_table(model_raw, MSMARCO_ID, "recall_1000")
            # DL 19
            scores["dl19_n"]  = self._get_score_for_disaggregated_table(model_raw, TREC19_ID, "ndcg_cut_10")
            scores["dl19_r"]  = self._get_score_for_disaggregated_table(model_raw, TREC19_ID, "recall_1000")
            # DL 20
            scores["dl20_n"]  = self._get_score_for_disaggregated_table(model_raw, TREC20_ID, "ndcg_cut_10")
            scores["dl20_r"]  = self._get_score_for_disaggregated_table(model_raw, TREC20_ID, "recall_1000")

            cache[model_raw] = scores

            # Update Max
            for k, v in scores.items():
                if v is not None and v > max_scores[k]:
                    max_scores[k] = v

        # 2. GENERATION PASS
        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")
        latex_lines.append(r"\caption{Evaluation results -- extended results for MS MARCO and TREC DL (2019 \& 2020). The best result per dataset is highlighted bold.}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(
            r"\textbf{Dataset $\rightarrow$} & "
            r"\multicolumn{2}{c}{\textbf{MS MARCO}} & "
            r"\multicolumn{2}{c}{\textbf{TREC DL 2019}} & "
            r"\multicolumn{2}{c}{\textbf{TREC DL 2020}} \\"
        )
        latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
        latex_lines.append(
            r"\textbf{Model $\downarrow$} & {\textbf{MRR}@10} & {\textbf{R}@1000} & "
            r"{\textbf{nDCG}@10} & {\textbf{R}@1000} & {\textbf{nDCG}@10} & {\textbf{R}@1000} \\"
        )
        latex_lines.append(r"\midrule")

        idx = 0
        for group_name, group_models in MODEL_FAMILIES_PRETTY.items():
            latex_lines.append(r"\multicolumn{7}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)
                s = cache[model_raw]

                vals = [
                    self._maybe_bold(s["msm_mrr"], max_scores["msm_mrr"], digit),
                    self._maybe_bold(s["msm_r"],   max_scores["msm_r"],   digit),
                    self._maybe_bold(s["dl19_n"],  max_scores["dl19_n"],  digit),
                    self._maybe_bold(s["dl19_r"],  max_scores["dl19_r"],  digit),
                    self._maybe_bold(s["dl20_n"],  max_scores["dl20_n"],  digit),
                    self._maybe_bold(s["dl20_r"],  max_scores["dl20_r"],  digit),
                ]

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            idx += 1
            if idx != len(ROW_BLOCKS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab-a:msmarco-trec-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def create_beir_table(self, digit: int = 3) -> str:
        # 1. PRE-CALCULATION PASS
        cache = {}
        max_scores = {d: -1.0 for d in BEIR_ORDER}
        all_models = [m for group in MODEL_FAMILIES_PRETTY.values() for m in group]

        for model_raw in all_models:
            scores = {}
            for d in BEIR_ORDER:
                if "cqa" not in d:
                    v = self._get_score_for_disaggregated_table(model_raw, d, "ndcg_cut_10")
                else:
                    cqa_items = [x for x in DATASETS if "cqa" in x]
                    cqa_vals = []
                    for i in cqa_items:
                        val = self._get_score_for_disaggregated_table(model_raw, i, "ndcg_cut_10")
                        if val is not None:
                            cqa_vals.append(val)
                    v = sum(cqa_vals) / len(cqa_vals) if cqa_vals else None
                
                scores[d] = v
                
                # Update Max
                if v is not None and v > max_scores[d]:
                    max_scores[d] = v
            
            cache[model_raw] = scores

        # 2. GENERATION PASS
        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\scriptsize")
        latex_lines.append(r"\caption{Evaluation results -- extended results for BEIR (nDCG$@10$). The best result per dataset is highlighted bold.}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{l" + "c"*len(BEIR_ORDER) + "}")
        latex_lines.append(r"\toprule")

        dataset_headers = " & ".join(
            r"\multirow{2}{*}{\textbf{" + BEIR_COL_NAMES[d] + r"}}"
            for d in BEIR_ORDER
        )
        latex_lines.append(r"\textbf{Dataset $\rightarrow$} & " + dataset_headers + r" \\")
        latex_lines.append(r"\textbf{Model $\downarrow$} & " + " & ".join([""] * len(BEIR_ORDER)) + r" \\")
        latex_lines.append(r"\midrule")

        idx = 0
        for group_name, group_models in MODEL_FAMILIES_PRETTY.items():
            latex_lines.append(r"\multicolumn{" + str(len(BEIR_ORDER)+1) + r"}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)
                row_scores = cache[model_raw]

                vals = []
                for d in BEIR_ORDER:
                    val_str = self._maybe_bold(row_scores[d], max_scores[d], digit)
                    vals.append(val_str)
                
                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            idx += 1
            if idx != len(ROW_BLOCKS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab-a:beir-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)

    # ---------------------------------------------------------
    # UPDATED: create_lotte_table
    # ---------------------------------------------------------
    def create_lotte_table(self, digit: int = 3) -> str:
        # 1. PRE-CALCULATION PASS
        cache = {}
        max_scores = {"search": -1.0, "forum": -1.0}
        all_models = [m for group in MODEL_FAMILIES_PRETTY.values() for m in group]

        for model_raw in all_models:
            s_search = self._get_score_for_disaggregated_table(model_raw, LOTTE_SEARCH_ID, "success_5")
            s_forum  = self._get_score_for_disaggregated_table(model_raw, LOTTE_FORUM_ID, "success_5")
            
            cache[model_raw] = {"search": s_search, "forum": s_forum}

            if s_search is not None and s_search > max_scores["search"]: max_scores["search"] = s_search
            if s_forum is not None and s_forum > max_scores["forum"]: max_scores["forum"] = s_forum

        # 2. GENERATION PASS
        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")
        latex_lines.append(r"\caption{Evaluation results -- extended results for LoTTE (Success$@5$). The best result per dataset is highlighted bold.}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"\textbf{Dataset $\rightarrow$} & \multicolumn{2}{c}{\textbf{Pooled}} \\")
        latex_lines.append(r"\cmidrule(lr){2-3}")
        latex_lines.append(r"\textbf{Model $\downarrow$} & {\textbf{Search}} & {\textbf{Forum}} \\")
        latex_lines.append(r"\midrule")
        
        idx = 0
        for group_name, group_models in MODEL_FAMILIES_PRETTY.items():
            latex_lines.append(r"\multicolumn{3}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)
                s = cache[model_raw]
                
                vals = [
                    self._maybe_bold(s["search"], max_scores["search"], digit),
                    self._maybe_bold(s["forum"],  max_scores["forum"],  digit)
                ]

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            idx += 1
            if idx != len(ROW_BLOCKS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab-a:lotte-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


    def create_instructir_followir_table(self, digit: int = 3) -> str:
        # 1. PRE-CALCULATION PASS
        keys = ["i_rob", "i_ndcg", "rob_map", "rob_p", "news_ndcg", "news_p", "core_map", "core_p"]
        max_scores = {k: -1.0 for k in keys}
        cache = {}
        all_models = [m for group in MODEL_FAMILIES_PRETTY.values() for m in group]

        for model_raw in all_models:
            # Helper to safely multiply by 100 only if not None
            def get_sc(ds_name, metric, scale=False):
                val = self._get_score_for_disaggregated_table(model_raw, ds_name, metric)
                if val is not None and scale: return val * 100
                return val

            s = {}
            s["i_rob"]     = get_sc(INSTRUCTIR_ID, "robustness_10")
            s["i_ndcg"]    = get_sc(INSTRUCTIR_ID, "ndcg_cut_10")
            s["rob_map"]   = get_sc(f"{FOLLOWIR_ROBUST_ID}_changed", "mean_avg_precision", True)
            s["rob_p"]     = get_sc(FOLLOWIR_ROBUST_ID, "p_mrr", True)
            s["news_ndcg"] = get_sc(f"{FOLLOWIR_NEWS21_ID}_changed", "ndcg_cut_5", True)
            s["news_p"]    = get_sc(FOLLOWIR_NEWS21_ID, "p_mrr", True)
            s["core_map"]  = get_sc(f"{FOLLOWIR_CORE17_ID}_changed", "mean_avg_precision", True)
            s["core_p"]    = get_sc(FOLLOWIR_CORE17_ID, "p_mrr", True)
            
            cache[model_raw] = s
            
            for k in keys:
                if s[k] is not None and s[k] > max_scores[k]:
                    max_scores[k] = s[k]

        # 2. GENERATION PASS
        latex_lines = []
        latex_lines.append(r"\begin{table}[h]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\scriptsize")
        latex_lines.append(r"\caption{Evaluation results -- extended results for \textsc{InstructIR} and FollowIR. The best result per dataset is highlighted bold.}")
        latex_lines.append(r"\vspace{5pt}")
        latex_lines.append(r"\begin{tabular}{lcccccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(
            r"\textbf{Dataset $\rightarrow$} & "
            r"\multicolumn{2}{c}{\textbf{\textsc{InstructIR}}} & "
            r"\multicolumn{6}{c}{\textbf{FollowIR}} \\"
        )
        latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-9}")
        latex_lines.append(
            r"& \multicolumn{2}{c}{\textbf{MS MARCO}} "
            r"& \multicolumn{2}{c}{\textbf{Robust04}} "
            r"& \multicolumn{2}{c}{\textbf{News21}} "
            r"& \multicolumn{2}{c}{\textbf{Core17}} \\"
        )
        latex_lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
        latex_lines.append(
            r"\textbf{Model $\downarrow$} & \textbf{Robustness}@10 & \textbf{nDCG}@10 & "
            r"\textbf{MAP} & $p$-\textbf{MRR} & \textbf{nDCG}@5 & $p$-\textbf{MRR} & "
            r"\textbf{MAP} & $p$-\textbf{MRR} \\"
        )
        latex_lines.append(r"\midrule")
        
        idx = 0
        for group_name, group_models in MODEL_FAMILIES_PRETTY.items():
            latex_lines.append(r"\multicolumn{9}{l}{\textbf{" + group_name + r"}}\\")
            latex_lines.append(r"\midrule")

            for model_raw in group_models:
                pretty_name = prettify_model_name(model_raw)
                s = cache[model_raw]

                vals = [
                    self._maybe_bold(s["i_rob"],     max_scores["i_rob"],     digit),
                    self._maybe_bold(s["i_ndcg"],    max_scores["i_ndcg"],    digit),
                    self._maybe_bold(s["rob_map"],   max_scores["rob_map"],   digit),
                    self._maybe_bold(s["rob_p"],     max_scores["rob_p"],     digit),
                    self._maybe_bold(s["news_ndcg"], max_scores["news_ndcg"], digit),
                    self._maybe_bold(s["news_p"],    max_scores["news_p"],    digit),
                    self._maybe_bold(s["core_map"],  max_scores["core_map"],  digit),
                    self._maybe_bold(s["core_p"],    max_scores["core_p"],    digit),
                ]

                latex_lines.append(pretty_name + " & " + " & ".join(vals) + r" \\")

            idx += 1
            if idx != len(ROW_BLOCKS) - 1:
                latex_lines.append(r"\midrule")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab-a:instructir-followir-results}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)