################################################################################
# hard_queries.py
#
# Description: ...
#
# Author: Valentin Velev
# Last updated: 31.01.2026
################################################################################

import json
import gzip
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_handler import DataHandler
from src.configs.datasets import DATASETS
from src.analysis.mappings import MODEL_NAMES_PRETTY, PRETTY_TO_FAMILY, FAMILY_COLORS


class HardQueries():
    """Determines hard queries based on raw rank of the relevant documents."""
    
    def __init__(self, data_handler: DataHandler):
        self.max_depth = 1000
        self.rel_min = 1
        self.data_handler = data_handler


    def _load_result_and_qrels(self, file_path: str, dataset_id: str):
        # Load results
        try:
            with gzip.open(f"{file_path}.json", "rt", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            try:
                with gzip.open(f"{file_path}.json.gz", "rt", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                print(f"{file_path} not found.")
                results = None

        # Choose variant (for FollowIR)
        if "jhu-clsp" in dataset_id:
            variant = "changed" # Take more representative variant
        else:
            variant = None

        # Load qrels
        _, _, qrels_iter = self.data_handler.read(dataset_id, variant=variant)
        qrels_df = pd.DataFrame(list(qrels_iter))

        return results, qrels_df


    def _build_qrels_map(self, qrels_df: pd.DataFrame, rel_min: int = 1):
        q_col = "qid"
        d_col = "docno"
        r_col = "label"

        rel_mask = qrels_df[r_col].astype(float) >= rel_min

        rel_df = qrels_df.loc[rel_mask, [q_col, d_col]].copy()
        rel_df[q_col] = rel_df[q_col].astype(str)
        rel_df[d_col] = rel_df[d_col].astype(str)

        qrels_map = (
            rel_df.groupby(q_col)[d_col]
                .apply(lambda s: set(s))
                .to_dict()
        )
        return qrels_map

    def _iter_doc_scores_for_q(self, results_for_q):
        if results_for_q is None:
            return

        # dict docid->score
        if isinstance(results_for_q, dict):
            for docid, score in results_for_q.items():
                try:
                    yield str(docid), float(score)
                except Exception:
                    continue
            return

        # list of pairs or list of dicts
        if isinstance(results_for_q, list):
            for item in results_for_q:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        yield str(item[0]), float(item[1])
                    except Exception:
                        continue
                elif isinstance(item, dict):
                    docid = item.get("doc_id") or item.get("docid") or item.get("docno") or item.get("id")
                    score = item.get("score") or item.get("sim") or item.get("value")
                    if docid is None or score is None:
                        continue
                    try:
                        yield str(docid), float(score)
                    except Exception:
                        continue
            return


    def _per_query_best_rel_rank_df(self, qrels_df: pd.DataFrame, results_run: dict, dataset_id: str, model_name: str) -> pd.DataFrame:
        """
        Returns DataFrame with columns: ["dataset","model","qid","score"]
        where score = best_rel_rank (1-based rank of the highest-ranked relevant doc).
        If no relevant doc in top self.max_depth, score = self.max_depth + 1.
        """
        if results_run is None:
            return pd.DataFrame(columns=["dataset", "model", "qid", "score"])

        qrels_map = self._build_qrels_map(qrels_df)

        rows = []
        for qid, res_q in results_run.items():
            qid_str = str(qid)
            rel_set = qrels_map.get(qid_str)
            if not rel_set:
                continue  # no judged relevant docs

            docs = list(self._iter_doc_scores_for_q(res_q))
            if not docs:
                rows.append({
                    "dataset": dataset_id,
                    "model": model_name,
                    "qid": qid_str,
                    "score": int(self.max_depth + 1),
                })
                continue

            ranked = sorted(docs, key=lambda x: x[1], reverse=True)[: self.max_depth]

            best_rank = None
            for i, (docid, _) in enumerate(ranked, start=1):
                if docid in rel_set:
                    best_rank = i
                    break

            if best_rank is None:
                best_rank = self.max_depth + 1

            rows.append({
                "dataset": dataset_id,
                "model": model_name,
                "qid": qid_str,
                "score": int(best_rank),
            })

        return pd.DataFrame(rows)


    def build_df_all(self) -> pd.DataFrame:
        dfs = []
        
        # 1. Generate all combinations first
        tasks = list(itertools.product(DATASETS, list(MODEL_NAMES_PRETTY.keys())))

        # 2. Iterate with a single progress bar
        for dataset_id, model_name in tqdm(tasks, desc="Processing All Pairs"):
            dataset_label = dataset_id.replace("/", "_").replace(":", "_")

            if "jhu-clsp" in dataset_label:
                file_path = f"outputs/results/{model_name}/{dataset_label}_changed"
            else:
                file_path = f"outputs/results/{model_name}/{dataset_label}"

            results, qrels_df = self._load_result_and_qrels(
                file_path=file_path,
                dataset_id=dataset_id,
            )

            if results is None: 
                continue

            df_pair = self._per_query_best_rel_rank_df(
                qrels_df=qrels_df,
                results_run=results,
                dataset_id=dataset_id,
                model_name=model_name,
            )

            dfs.append(df_pair)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["dataset","model","qid","score"])
        return df


    def select_bottom_percent_per_dataset_model(self, df: pd.DataFrame, p: float = 0.1):
        """
        Selects the bottom p% queries per (dataset, model) based on score.
        """
        def select_group(g: pd.DataFrame) -> pd.DataFrame:
            N = len(g)
            if N == 0:
                return g.copy()

            K = int(np.ceil(p * N))

            # Sort by hardness: higher rank = harder
            g_sorted = g.sort_values(
                by=["score", "qid"],
                ascending=[False, True],
            )

            return g_sorted.head(K)

        out = (
            df.groupby(["dataset", "model"], group_keys=False)
            .apply(select_group)
            .reset_index(drop=True)
        )

        return out
    
    def jaccard_similarity(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 1.0


    def jaccard_distance(self, a: set, b: set) -> float:
        return 1.0 - self.jaccard_similarity(a, b)


    def hard_sets_from_selected_df(self, selected_df: pd.DataFrame) -> dict:
        """
        Build hard_sets dict from the output of select_bottom_percent_per_dataset_model().
        """
        required = {"dataset", "model", "qid"}
        missing = required - set(selected_df.columns)
        if missing:
            raise ValueError(f"selected_df missing required columns: {missing}")

        tmp = selected_df.copy()
        tmp["qid"] = tmp["qid"].astype(str)

        hard_sets = {
            (ds, m): set(g["qid"].tolist())
            for (ds, m), g in tmp.groupby(["benchmark", "model"])
        }
        return hard_sets


    def ordered_models_for_dataset(self, selected_df: pd.DataFrame, dataset_id: str, name_map: dict) -> list[str]:
        present = selected_df.loc[selected_df["benchmark"] == dataset_id, "model"].astype(str).unique().tolist()

        # keep mapping order first (only those present)
        ordered = [m for m in name_map.keys() if m in present]

        # append any "unknown" models not in the mapping (sorted for stability)
        unknown = sorted([m for m in present if m not in name_map])
        return ordered + unknown


    def jaccard_matrix_for_dataset_from_selected(self, selected_df: pd.DataFrame, dataset_id: str, use_distance: bool = True) -> pd.DataFrame:
        """
        Compute Jaccard similarity/distance matrix over models for a dataset, using the hard queries DataFrame.
        """
        hard_sets = self.hard_sets_from_selected_df(selected_df)

        # models present for this dataset in the selected_df
        models = self.ordered_models_for_dataset(selected_df, dataset_id, MODEL_NAMES_PRETTY)
        pretty = {m: MODEL_NAMES_PRETTY.get(m, m) for m in models}
        
        n = len(models)

        mat = np.zeros((n, n), dtype=float)
        for i, m1 in enumerate(models):
            s1 = hard_sets.get((dataset_id, m1), set())
            for j, m2 in enumerate(models):
                s2 = hard_sets.get((dataset_id, m2), set())
                mat[i, j] = self.jaccard_distance(s1, s2) if use_distance else self.jaccard_similarity(s1, s2)

        mat_df = pd.DataFrame(mat, index=models, columns=models)
        mat_df = mat_df.rename(index=pretty, columns=pretty)

        return mat_df


    def swap_models(self, mat_df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
        """
        Swap the positions of two models (labels) in both rows and columns.
        """
        order = list(mat_df.index)
        ia, ib = order.index(a), order.index(b)
        order[ia], order[ib] = order[ib], order[ia]
        return mat_df.loc[order, order]
    
    
    def plot_jaccard_heatmap_ax(
        self,
        mat_df: pd.DataFrame,
        ax: plt.Axes,
        title: str,
        use_distance: bool = True,
        show_cbar: bool = False,
        cbar_ax = None,
        cbar_orientation = "vertical",
        title_size = 20,
        tick_size = 10,
        val_lab_size = 5,
        cbar_shrink = None,
        cbar_title_size = 20,
        cbar_tick_size = 16
    ):
        """
        ...
        """
        hm = sns.heatmap(
            mat_df,
            ax=ax,
            vmin=0, vmax=1,
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": val_lab_size},
            cmap="viridis",
            cbar=show_cbar,
            cbar_ax=cbar_ax,
            cbar_kws={
                "label": "Jaccard distance" if use_distance else "Jaccard similarity", 
                "orientation": cbar_orientation,
                "shrink": cbar_shrink
            },
        )

        # color x-axis model labels
        for label in ax.get_xticklabels():
            pretty_name = label.get_text()
            family = PRETTY_TO_FAMILY.get(pretty_name)
            if family is not None:
                label.set_color(FAMILY_COLORS[family])
                label.set_fontweight("bold")

        # color y-axis model labels
        for label in ax.get_yticklabels():
            pretty_name = label.get_text()
            family = PRETTY_TO_FAMILY.get(pretty_name)
            if family is not None:
                label.set_color(FAMILY_COLORS[family])
                label.set_fontweight("bold")

        ax.tick_params(axis="x", labelsize=tick_size)
        ax.tick_params(axis="y", labelsize=tick_size)

        ax.set_title(title, fontsize=title_size, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # If a colorbar is shown, style its label
        if show_cbar:
            cbar = hm.collections[0].colorbar
            cbar.set_label(
                "Jaccard distance" if use_distance else "Jaccard similarity",
                fontsize=cbar_title_size,
                weight="bold"
            )
            cbar.ax.tick_params(labelsize=cbar_tick_size)
            cbar.ax.yaxis.labelpad = 20