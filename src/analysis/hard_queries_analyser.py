import os
import json
import gzip
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from src.evaluator import Evaluator


class HardQueriesAnalyser():
    def __init__(self, results_root: str):
        self.results_root = results_root

    def _dataset_to_label(self, dataset: str) -> str:
        return dataset.replace("/", "_").replace(":", "_")

    def _load_results_for_model(self, model: str, dataset: str):
        """
        Load the results JSON for a (model, dataset) pair.
        Returns the 'results' dict[qid -> {docid: score}] or None if missing.
        """
        dataset_label = self._dataset_to_label(dataset)
        path = os.path.join(self.results_root, model, f"{dataset_label}.json")
        if not os.path.exists(path):
            print(f"[WARN] Missing results for model={model}, dataset={dataset}: {path}")
            return None

        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                results = json.load(f)
        except:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)

        return results

    def _resolve_models_arg(self, models, dataset: str) -> list[str]:
        """
        If `models` is 'all', return all model names (subdirs of results_root)
        that have a results file for this dataset. Otherwise, return `models` unchanged.
        """
        if isinstance(models, str) and models.lower() == "all":
            dataset_label = self._dataset_to_label(dataset)
            all_models = []
            for name in os.listdir(self.results_root):
                model_dir = os.path.join(self.results_root, name)
                if not os.path.isdir(model_dir):
                    continue
                result_path = os.path.join(model_dir, f"{dataset_label}.json")
                if os.path.exists(result_path):
                    all_models.append(name)
            print(f"[INFO] Using all models with results for dataset '{dataset}': {all_models}")
            return all_models
        # assume it's already a list-like
        return list(models)

    def _get_per_query_metric_for_models(
        self,
        models,
        dataset: str,
        handler
    ) -> tuple[pd.DataFrame, str]:
        """
        For a given dataset and models (list or 'all'), compute per-query metrics
        (one scalar metric per model) using your Evaluator.

        Returns:
        - perq_all_df: DataFrame indexed by qid, columns = model names, values = metric
        - metric_name: the metric used (e.g., 'ndcg_cut_10', 'recip_rank_10', ...)
        """
        # Resolve "all" to the available models for this dataset
        models = self._resolve_models_arg(models, dataset)

        # Load qrels once
        _, _, qrels_iter = handler.read(dataset, variant=None)
        qrels_df = pd.DataFrame(list(qrels_iter))

        evaluator = Evaluator(dataset, skip_self_matches="auto")

        perq_all = {}
        metric_name = None

        for model in models:
            results = self._load_results_for_model(model, dataset)
            if results is None:
                continue  # skip missing

            metrics_agg, metrics_perq, summary_stats = evaluator.evaluate(qrels_df, results, print_msg=False)
            perq_df = pd.DataFrame.from_dict(metrics_perq, orient="index")
            perq_df.index.name = "qid"

            if perq_df.empty:
                print(f"[WARN] No per-query metrics for model={model}, dataset={dataset}")
                continue

            # Decide the metric name once using Evaluator's mapping
            if metric_name is None:
                for m in evaluator.measure_objs:
                    mname = evaluator.metric_name_map.get(m, str(m))
                    if mname in perq_df.columns:
                        metric_name = mname
                        break
                if metric_name is None:
                    metric_name = perq_df.columns[0]
                print(f"[INFO] Using metric '{metric_name}' for dataset '{dataset}'")

            if metric_name not in perq_df.columns:
                print(f"[WARN] Metric '{metric_name}' not in per-query metrics for model={model}. Skipping.")
                continue

            s = perq_df[metric_name].astype(float)
            s.name = model
            perq_all[model] = s

        if not perq_all:
            raise ValueError(f"No per-query metrics found for dataset={dataset} across models={models}")

        perq_all_df = pd.concat(perq_all.values(), axis=1)
        perq_all_df.index.name = "qid"

        return perq_all_df, metric_name

    def find_hard_queries_across_models(
        self,
        models,
        dataset: str,
        handler,
        p: float = 0.6,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        A query is 'hard' if > p fraction of models have metric < threshold.

        `models` can be:
        - a list of model names, or
        - the string "all" to use all models that have results for this dataset.

        Returns a DataFrame with:
        qid, text, frac_below_threshold, and one column per model with the metric.
        """
        perq_all_df, metric_name = self._get_per_query_metric_for_models(
            models=models,
            dataset=dataset,
            handler=handler
        )

        # Drop queries where the metric is missing for all models
        perq_all_df = perq_all_df.dropna(how="all")

        # Fraction of models below the threshold per query
        below = (perq_all_df < threshold).astype(float)
        frac_below = below.mean(axis=1)

        hard_mask = frac_below > p
        hard_qids = perq_all_df.index[hard_mask]

        hard_df = perq_all_df.loc[hard_qids].copy()
        hard_df["frac_below_threshold"] = frac_below.loc[hard_qids]

        # Attach query text
        _, queries_iter, _ = handler.read(dataset, variant=None)
        queries_df = pd.DataFrame(list(queries_iter))
        if "query_id" in queries_df.columns:
            queries_df = queries_df.rename(columns={"query_id": "qid"})
        queries_df["qid"] = queries_df["qid"].astype(str)

        hard_df = hard_df.reset_index().merge(queries_df, on="qid", how="left")

        # Column order: qid, text, frac_below_threshold, then per-model metrics
        model_names = [m for m in perq_all_df.columns if m in hard_df.columns]
        cols = ["qid", "text", "frac_below_threshold"] + model_names
        hard_df = hard_df[cols]

        print(
            f"[INFO] Dataset '{dataset}': using metric '{metric_name}', "
            f"{len(hard_df)} hard queries at p={p}, threshold={threshold}"
        )

        return hard_df


    def topic_model_hard_queries(self, df, min_cluster_size: int = 7, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        texts = df["text"].to_list()

        # BERTopic model
        umap_model = UMAP(
            n_neighbors=15, # roughly controls the number of clusters
            n_components=10,
            min_dist=0.0, 
            metric='cosine',
            random_state=42
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, # roughly controls the number of clusters
            min_samples=1, 
            metric='euclidean',
            prediction_data=True
        )

        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            embedding_model=embedding_model,
            nr_topics="auto",
            top_n_words=10,
            calculate_probabilities=True,
            verbose=True
        )

        # Embeddings
        embed_model = SentenceTransformer(embedding_model)
        embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        
        # Fit and transform
        topics, probs = topic_model.fit_transform(texts, embeddings)

        # Add topic column
        df = df.copy()
        df["topic_id"] = topics

        return df, topic_model, embeddings
