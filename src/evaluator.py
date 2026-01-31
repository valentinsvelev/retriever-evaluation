################################################################################
# Title
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import math
import pandas as pd
import numpy as np
import ir_measures
from ir_measures import calc_aggregate, AP, RR, R, nDCG, Success, iter_calc, MAP, MRR
from typing import Dict, Optional, Tuple


class Evaluator:
    """
    Handles the evaluation of retrieval results using ir_measures.
    """

    def __init__(self, dataset_name, skip_self_matches="auto"):
        """
        skip_self_matches: "auto" | "always" | "never"
            - "auto": skip qid==docid only if qrels say it's NOT relevant (safe default)
            - "always": always drop qid==docid from results
            - "never": keep everything
        """
        self.dataset_name = dataset_name.lower()
        self.skip_self_matches = skip_self_matches
        
        self.compute_robustness = ("kaist" in self.dataset_name)

        if "trec-dl" in self.dataset_name:
            self.measure_objs = [nDCG@10, RR, MAP, R@1000]
            self.metric_name_map = {(nDCG@10): "ndcg_cut_10", (RR): "recip_rank", (MAP): "mean_avg_precision", (R@1000): "recall_1000"}
        elif "msmarco" in self.dataset_name:
            self.measure_objs = [MRR@10, R@1000]
            self.metric_name_map = {(MRR@10): "recip_rank_10", (R@1000): "recall_1000"}
        elif "lotte" in self.dataset_name:
            self.measure_objs = [Success@5]
            self.metric_name_map = {(Success@5): "success_5"}
        elif "kaist" in self.dataset_name:
            self.measure_objs = [nDCG@10]
            self.metric_name_map = {(nDCG@10): "ndcg_cut_10"}
        elif "jhu-clsp/robust04-instructions" in self.dataset_name or "jhu-clsp/core17-instructions" in self.dataset_name:
            self.measure_objs = [MAP@1000]
            self.metric_name_map = {(MAP@1000): "mean_avg_precision_1000"}
        elif "jhu-clsp/news21-instructions" in self.dataset_name:
            self.measure_objs = [nDCG@5]
            self.metric_name_map = {(nDCG@5): "ndcg_cut_5"}
        else: # BEIR datasets
            self.measure_objs = [nDCG@10, R@100]
            self.metric_name_map = {(nDCG@10): "ndcg_cut_10", (R@100): "recall_100"}


    def _bootstrap_mean_stats(self, values, n_boot=100000, ci=0.95, random_state=42):
        """
        Returns both per-query SD and bootstrap-based SE/CI for the mean.
        """
        s = pd.Series(values).dropna().to_numpy()
        n = s.shape[0]
        if n == 0:
            return {k: float("nan") for k in ["standard_dev", "standard_error", "ci95_low", "ci95_high"]}
        if n == 1:
            v = float(s[0])
            return {
                "standard_dev": 0.0,
                "standard_error": 0.0,
                "ci95_low": v,
                "ci95_high": v,
            }

        # --- per-query variability
        sd = float(s.std(ddof=1))

        # --- bootstrap mean distribution
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = s[idx].mean(axis=1)

        se = float(boot_means.std(ddof=1))
        alpha = (1.0 - ci) / 2.0
        low = float(np.quantile(boot_means, alpha))
        high = float(np.quantile(boot_means, 1.0 - alpha))

        return {
            "standard_dev": sd,       # across queries
            "standard_error": se,     # of the mean
            "ci95_low": low,
            "ci95_high": high,
        }


    @staticmethod
    def _base_qid(qid: str) -> str:
        """
        Base query identifier for robustness grouping.
        """
        return qid.split("_")[0]


    def p_mrr(
        self,
        qrels_og_df: pd.DataFrame,
        qrels_changed_df: pd.DataFrame,
        run_og: dict[str, dict[str, float]],
        run_changed: dict[str, dict[str, float]],
        k: int | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute FollowIR p-MRR (pairwise MRR) between an original and changed run.
        Returns (macro_avg_p_mrr, per_query_p_mrr).
        """

        # --- normalize inputs
        qrels_og = qrels_og_df.copy()
        qrels_og["qid"] = qrels_og["qid"].astype(str)
        qrels_og["docno"] = qrels_og["docno"].astype(str)
        qrels_og["label"] = qrels_og["label"].astype(int)

        qrels_ch = qrels_changed_df.copy()
        qrels_ch["qid"] = qrels_ch["qid"].astype(str)
        qrels_ch["docno"] = qrels_ch["docno"].astype(str)
        qrels_ch["label"] = qrels_ch["label"].astype(int)

        # --- set of docs whose relevance changed: label_og=1 -> label_new=0
        rel_og = qrels_og[qrels_og["label"] > 0][["qid", "docno"]].drop_duplicates()
        rel_new = qrels_ch[qrels_ch["label"] > 0][["qid", "docno"]].drop_duplicates()

        # docs that were relevant before but are NOT relevant after
        changed = (
            rel_og.merge(rel_new, on=["qid", "docno"], how="left", indicator=True)
                .loc[lambda df: df["_merge"] == "left_only", ["qid", "docno"]]
        )

        # --- build rank maps for both runs
        def ranks_for_run(run: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
            out: dict[str, dict[str, int]] = {}
            for qid, doc2score in run.items():
                # sort by score DESC
                items = sorted(doc2score.items(), key=lambda kv: kv[1], reverse=True)
                if k is not None:
                    items = items[:k]
                rank_map = {docid: (i + 1) for i, (docid, _) in enumerate(items)}
                out[str(qid)] = rank_map
            return out

        ranks_og = ranks_for_run(run_og)
        ranks_new = ranks_for_run(run_changed)

        # --- per-query aggregation
        perq: dict[str, float] = {}
        for qid, group in changed.groupby("qid"):
            qid = str(qid)
            changed_docs = group["docno"].tolist()

            # If a query has no changed docs, skip it (no contribution to macro avg)
            if not changed_docs:
                continue

            # Build RR maps: RR=1/rank, RR=0 if missing or rank>k (if k given)
            rr_og = {}
            rr_new = {}
            for d in changed_docs:
                r_og = ranks_og.get(qid, {}).get(d, None)
                r_new = ranks_new.get(qid, {}).get(d, None)

                # apply cutoff if given (treat beyond k as not retrieved)
                if k is not None:
                    if r_og is None or r_og > k:
                        r_og = None
                    if r_new is None or r_new > k:
                        r_new = None

                rr_og[d] = 0.0 if r_og is None else 1.0 / float(r_og)
                rr_new[d] = 0.0 if r_new is None else 1.0 / float(r_new)

            # p-MRR per changed doc using the paper's piecewise rule
            vals = []
            for d in changed_docs:
                RRog = rr_og[d]
                RRnew = rr_new[d]

                # If doc wasn't retrieved originally (RRog==0), the ratio is undefined.
                # FollowIR's reranking setup ensures original relevant docs are in the pool,
                # so this should be rare in full-retrieval. We skip such docs.
                if RRog == 0.0:
                    continue

                # Decide branch using ranks (more stable than comparing RR floats)
                r_og = 1.0 / RRog
                r_new = float("inf") if RRnew == 0.0 else 1.0 / RRnew

                if r_og > r_new:
                    # moved UP after change (bad; doc should be demoted) -> negative value
                    val = (RRog / RRnew) - 1.0 if RRnew > 0.0 else -1.0  # guard div-by-zero
                else:
                    # moved DOWN / stayed same after change (good) -> positive or zero
                    val = 1.0 - (RRnew / RRog)
                # Clip to [-1, 1] to be safe with numerical noise
                vals.append(float(max(-1.0, min(1.0, val))))

            if len(vals) == 0:
                continue
            perq[qid] = float(np.mean(vals))

        macro = float(np.mean(list(perq.values()))) if perq else float("nan")
        return macro, perq


    def evaluate(self, qrels_df: pd.DataFrame, results: dict, print_msg: bool = True):
        if print_msg:
            print("Evaluating with ir_measures...")

        # --- Build qrels lookup + ir_measures-friendly qrels ---
        qrels_df = qrels_df.copy()
        qrels_df["qid"] = qrels_df["qid"].astype(str)
        qrels_df["docno"] = qrels_df["docno"].astype(str)
        qrels_df["label"] = qrels_df["label"].astype(int)

        qrels_lookup = {}
        for _, row in qrels_df.iterrows():
            qrels_lookup.setdefault(row["qid"], {})[row["docno"]] = row["label"]

        qrels = {qid: dict(docmap) for qid, docmap in qrels_lookup.items()}

        # --- Normalize run keys + apply skip_self_matches ---
        results_str = {
            str(q): {str(d): float(s) for d, s in doc_scores.items()}
            for q, doc_scores in results.items()
        }

        run = {}
        for qid in qrels.keys():
            q_run = dict(results_str.get(qid, {}))

            if qid in q_run:
                if self.skip_self_matches == "always":
                    q_run.pop(qid, None)
                elif self.skip_self_matches == "auto":
                    if qrels_lookup.get(qid, {}).get(qid, 0) == 0:
                        q_run.pop(qid, None)

            run[qid] = q_run

        # Per query
        perq = {}
        for rec in iter_calc(self.measure_objs, qrels, run):
            qid = str(rec.query_id)
            name = self.metric_name_map.get(rec.measure, str(rec.measure))
            perq.setdefault(qid, {})[name] = float(rec.value)

        # Aggregate
        agg = calc_aggregate(self.measure_objs, qrels, run)
        aggregated_scores = { self.metric_name_map[m]: float(agg.get(m, 0.0)) for m in self.measure_objs }

        # --- Uncertainty stats across queries ---
        perq_df = pd.DataFrame.from_dict(perq, orient="index") if perq else pd.DataFrame()

        summary_stats = {}
        for m in self.measure_objs:
            mname = self.metric_name_map[m]
            if mname in perq_df.columns:
                s = perq_df[mname].dropna()
                stats = self._bootstrap_mean_stats(s)
                summary_stats[mname] = stats
            else:
                summary_stats[mname] = {
                    "standard_dev": float("nan"),
                    "standard_error": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                }

        # --- Robustness@10 for InstructIR: mean of per-base min nDCG@10 across instruction variants ---
        if self.compute_robustness and ("ndcg_cut_10" in perq_df.columns) and (not perq_df.empty):
            # Map qid -> base_qid
            base_ids = perq_df.index.to_series().map(self._base_qid)
            ndcg = perq_df["ndcg_cut_10"].astype(float)

            # Build a DataFrame with base ids, take per-base minimum nDCG@10
            tmp = pd.DataFrame({"base_qid": base_ids.values, "ndcg10": ndcg.values}).dropna()
            per_base_min = tmp.groupby("base_qid", as_index=True)["ndcg10"].min()

            # Aggregate robustness@10 = mean of per-base minima
            robustness_10 = float(per_base_min.mean()) if len(per_base_min) > 0 else float("nan")
            aggregated_scores["robustness_10"] = robustness_10

            # Uncertainty over base queries
            summary_stats["robustness_10"] = self._bootstrap_mean_stats(per_base_min.values)

        return aggregated_scores, perq, summary_stats
