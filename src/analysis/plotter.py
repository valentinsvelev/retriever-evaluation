import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from src.analysis.mappings import MODEL_NAMES_PRETTY, model_to_family

plt.rcParams["font.family"] = "serif"


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

class Plotter():
    def _plot_reg_line(self, df: pd.DataFrame, col_x: str, col_y: str, reg_type: str = "ols") -> None:
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[col_x, col_y])
        df = df[df[col_x] > 0]

        if len(df) < 3:
            print("Not enough valid points for regression.")
            return
        
        X = np.log(df[col_x].values).reshape(-1, 1)
        y = df[col_y].values
        x_line = np.linspace(df[col_x].min(), df[col_x].max(), 300)
        log_x_line = np.log(x_line)

        if reg_type == "ols":
            a_ols, b_ols = np.polyfit(X.flatten(), y, 1)
            y_line_ols = a_ols * log_x_line + b_ols

            # --- Compute OLS confidence band ---
            y_pred_ols = a_ols * X.flatten() + b_ols
            residuals = y - y_pred_ols
            dof = len(X) - 2
            stderr = np.sqrt(np.sum(residuals**2) / dof)

            ols_upper = y_line_ols + 1.96 * stderr
            ols_lower = y_line_ols - 1.96 * stderr

            # Plot OLS line + band
            plt.plot(x_line, y_line_ols, color="blue", linewidth=2, alpha=0.9, label="OLS")
            plt.fill_between(x_line, ols_lower, ols_upper, color="blue", alpha=0.12)


        if reg_type == "huber":
            huber = HuberRegressor().fit(X, y)
            a_h = huber.coef_[0]
            b_h = huber.intercept_
            y_line_huber = a_h * log_x_line + b_h

            # --- Compute Huber confidence band ---
            mask = huber.outliers_ == 0 if hasattr(huber, "outliers_") else np.ones(len(y), dtype=bool)
            inliers_X = X[mask].flatten()
            inliers_y = y[mask]
            inlier_pred = a_h * inliers_X + b_h
            residuals_h = inliers_y - inlier_pred

            dof_h = max(len(inliers_X) - 2, 1)
            stderr_h = np.sqrt(np.sum(residuals_h**2) / dof_h)

            huber_upper = y_line_huber + 1.96 * stderr_h
            huber_lower = y_line_huber - 1.96 * stderr_h

            # Plot Huber line + band
            plt.plot(x_line, y_line_huber, color="red", linestyle="--", linewidth=2, alpha=0.9, label="Huber")
            plt.fill_between(x_line, huber_lower, huber_upper, color="red", alpha=0.10)


        if reg_type == "ransac":
            ransac = RANSACRegressor(LinearRegression()).fit(X, y)
            a_r = ransac.estimator_.coef_[0]
            b_r = ransac.estimator_.intercept_

            y_line_ransac = a_r * log_x_line + b_r

            # --- Compute RANSAC confidence band ---
            inlier_mask = ransac.inlier_mask_
            inliers_X_r = X[inlier_mask].flatten()
            inliers_y_r = y[inlier_mask]

            pred_inliers = a_r * inliers_X_r + b_r
            residuals_r = inliers_y_r - pred_inliers

            dof_r = max(len(inliers_X_r) - 2, 1)
            stderr_r = np.sqrt(np.sum(residuals_r**2) / dof_r)

            ransac_upper = y_line_ransac + 1.96 * stderr_r
            ransac_lower = y_line_ransac - 1.96 * stderr_r

            # Plot RANSAC line + band
            plt.plot(x_line, y_line_ransac, color="green", linestyle=":", linewidth=2, alpha=0.9, label="RANSAC")
            plt.fill_between(x_line, ransac_lower, ransac_upper, color="green", alpha=0.10)


    def plot(self, plot_type: str, df: pd.DataFrame, x: str, y: str, width: int, height: int, xlab: str, ylab: str, log: list[bool], reg_type = None, show: bool = True, save_path: str = "path") -> None:
        plt.figure(figsize=(width, height))

        if plot_type == "runtimeXdatsize":
            df["family"] = df["model"].map(model_to_family)

            # Unique models
            models = df["model"].unique()

            # 1) Build a color palette for the models (automatic, no manual dict needed)
            palette = sns.color_palette("husl", n_colors=len(models))
            MODEL_COLORS = dict(zip(models, palette))

            # 2) Plot, using the explicit palette and family markers
            ax = sns.lineplot(
                data=df,
                x=x,
                y=y,
                hue="model",
                style="family",
                markers=FAMILY_MARKERS,   # your family -> marker dict
                palette=MODEL_COLORS,     # <-- explicit mapping
                legend=False,             # we'll build our own legend
                alpha=0.5,
            )

            # 3) Build combined legend: correct color (per model) + marker (per family)
            legend_elements = []
            for model in models:
                pretty_label = MODEL_NAMES_PRETTY.get(model, model)
                family = model_to_family[model]      # your model -> family dict
                marker = FAMILY_MARKERS[family]
                color = MODEL_COLORS[model]

                legend_elements.append(
                    Line2D(
                        [0], [0],
                        label=pretty_label,
                        color=color,
                        marker=marker,
                        linewidth=2,
                        markersize=6,
                        markerfacecolor=color,
                        markeredgecolor=color,
                    )
                )

            # 4) Add legend
            legend = ax.legend(
                handles=legend_elements,
                title="Retrievers",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=4,
                frameon=False,
            )

            legend.get_title().set_fontweight("bold")

        elif plot_type == "forest":
            # Plot
            for i, row in df.iterrows():
                model_name = row["model"]
                family = df.loc[df["model"] == model_name, "family"].iloc[0]
                color = FAMILY_COLORS.get(family, "gray")

                plt.errorbar(
                    x=row['mean'],
                    y=[i],
                    xerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]],
                    fmt='o',
                    capsize=4,
                    color=color,
                    markersize=8
                )

            plt.yticks(
                ticks=range(len(df)),
                labels=df["model"].apply(lambda m: MODEL_NAMES_PRETTY.get(m, m))
            )
            
            # Legend
            handles = [
                plt.Line2D([0], [0], color=c, marker='o', linestyle='', label=fam.replace("_", " ").title())
                for fam, c in FAMILY_COLORS.items()
            ]
            legend = plt.legend(handles=handles, title="Model Family", loc="lower right")
            legend.get_title().set_fontweight("bold")

        elif plot_type == "scatter":
            # Plot
            for _, row in df.iterrows():
                family = row["family"]
                color = FAMILY_COLORS.get(family, "gray")
                marker = FAMILY_MARKERS.get(family, "x")

                plt.scatter(
                    row[x],
                    row[y],
                    s=120,
                    c=color,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=1,
                )

                pretty_name = MODEL_NAMES_PRETTY.get(row["model"], row["model"])

                # label next to point
                plt.text(
                    row[x] * 1.02,
                    row[y] + 0.02,
                    pretty_name,
                    fontsize=8,
                )

            plt.ylim(top=1)
            plt.ylim(bottom=0)
            
            # Legend
            handles = []
            for fam, color in FAMILY_COLORS.items():
                if fam == "other":
                    continue
                marker = FAMILY_MARKERS[fam]
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker=marker,
                        markersize=8,
                        linestyle="",
                        markerfacecolor=color,
                        markeredgecolor="black",
                        label=fam.replace("_", " ").title(),
                    )
                )
            legend = plt.legend(
                handles=handles,
                title="Model Family",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5), # position relative to the axes
                borderaxespad=0
            )
            legend.get_title().set_fontweight("bold")

        elif plot_type == "sina":
            # Plot
            ## 1. smooth violin density per family
            sns.violinplot(
                data=df,
                x=x,
                y=y,
                palette=FAMILY_COLORS,
                inner=None,
                cut=0,
                alpha=0.15
            )

            ## 2. jittered raw points
            sns.stripplot(
                data=df,
                x=x,
                y=y,
                color="black",
                alpha=1,
                jitter=True,
                size=5,
            )

            ## 3. mean lines
            means = df.groupby(x)[y].mean()
            
            for i, (cat, mean_val) in enumerate(means.items()):
                plt.hlines(
                    y=mean_val,
                    xmin=i - 0.4,
                    xmax=i + 0.4,
                    colors="red",
                    linewidth=3
                )

            # Legend

        else:
            print("Plot type not supported.")
            return None

        plt.xlabel(xlab, weight="bold")
        plt.ylabel(ylab, weight="bold")
        if log[0]:
            plt.xscale("log")
        if log[1]:
            plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)

        if reg_type != None:
            if reg_type == "ols":
                self._plot_reg_line(df, x, y, reg_type)
            elif reg_type == "huber":
                self._plot_reg_line(df, x, y, reg_type)
            elif reg_type == "ransac":
                self._plot_reg_line(df, x, y, reg_type)

        plt.tight_layout()

        if show:
            plt.show()
        
        if isinstance(save_path, str):
            plt.savefig(f"plots/{save_path}.pdf", dpi=1000)