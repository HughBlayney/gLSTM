import json
import os
import pickle as pkl
from dataclasses import asdict
from itertools import product
from statistics import mean, stdev
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPSILON = 1e-10
MODEL_NAME = "gLSTM"
get_model_name = lambda model_name: (
    "gLSTM" if "xlstm" in model_name.lower() else model_name
)

# Text width is 5.5in - let's say we have 5in of space to play with. Plan
# accordingly
TEXT_WIDTH = 6
STANDARD_FIGSIZE = (4, 1.75)
BIG_FIGSIZE = (TEXT_WIDTH, 3)
BIG_SHORT_FIGSIZE = (BIG_FIGSIZE[0], STANDARD_FIGSIZE[1])
HALF_FIGSIZE = (TEXT_WIDTH / 2, STANDARD_FIGSIZE[1])
PARAM_FIGSIZE = (TEXT_WIDTH - STANDARD_FIGSIZE[0], STANDARD_FIGSIZE[1])
MAKE_STANDARD_LEGEND = lambda axes, ncols: axes.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=ncols,
    columnspacing=0.8,
    handlelength=1,
)
MAKE_RIGHT_LEGEND = lambda axes, ncols: axes.legend(
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=ncols,
    columnspacing=0.8,
    handlelength=1,
)
PLOT_KWARGS = {
    "markersize": 5,
    "linewidth": 2,
}


def results_dirs_to_df(
    keywords_to_filepath: Callable,
    metric_name: Union[str, Callable],
    goal: Callable,
    num_repeats: int = 4,
    **kwargs,
):
    keywords = list(kwargs.keys())
    values = [kwargs[k] for k in keywords]

    records = []
    failed_value_combinations = []

    for value_combination in product(*values):
        result_dir = keywords_to_filepath(*value_combination)
        if not isinstance(metric_name, str):
            metric_str = metric_name(*value_combination)
        else:
            metric_str = metric_name

        metric_goal = goal(*value_combination)
        for run in range(num_repeats):
            try:
                # Load validation APs
                val_path = os.path.join(result_dir, str(run), "val", "stats.json")
                test_path = os.path.join(result_dir, str(run), "test", "stats.json")

                logging_path = os.path.join(result_dir, str(run), "logging.log")
                # Verify that the logging file contains the string "Task done", if not,
                # skip loading these results - it's not a valid run
                with open(logging_path) as f:
                    if "Task done" not in f.read():
                        raise ValueError(f"Task not done for {result_dir}, run {run}")

                with open(val_path) as f_val, open(test_path) as f_test:
                    val_results = [json.loads(result_line) for result_line in f_val]
                    test_results = [json.loads(result_line) for result_line in f_test]

                    # Get last epoch with best metric
                    best_metric = metric_goal(val_results, key=lambda x: x[metric_str])[
                        metric_str
                    ]
                    best_epoch = [
                        x["epoch"] for x in val_results if x[metric_str] == best_metric
                    ][-1]

                    # Get test metric at that epoch
                    test_values_at_best_epoch = next(
                        x for x in test_results if x["epoch"] == best_epoch
                    )
                    record = {
                        keyword: value
                        for keyword, value in zip(keywords, value_combination)
                    }
                    record.update(test_values_at_best_epoch)
                    records.append(record)
            except Exception as e:
                failed_value_combinations.append((value_combination, e))

    return pd.DataFrame(records), failed_value_combinations


def results_dirs_to_separated_dfs(keywords_to_filepath: Callable, **kwargs):
    keywords = list(kwargs.keys())
    values = [kwargs[k] for k in keywords]

    failed_value_combinations = []

    # Always train, val, test
    value_combination_to_dfs: dict[tuple[str], list[tuple[pd.DataFrame]]] = {}

    for value_combination in product(*values):
        result_dir = keywords_to_filepath(*value_combination)

        value_combination_to_dfs[value_combination] = []
        for run in range(4):
            try:
                # Load validation APs
                train_path = os.path.join(result_dir, str(run), "train", "stats.json")
                val_path = os.path.join(result_dir, str(run), "val", "stats.json")
                test_path = os.path.join(result_dir, str(run), "test", "stats.json")

                logging_path = os.path.join(result_dir, str(run), "logging.log")
                # Verify that the logging file contains the string "Task done", if not,
                # skip loading these results - it's not a valid run
                with open(logging_path) as f:
                    if "Task done" not in f.read():
                        raise ValueError(f"Task not done for {result_dir}, run {run}")

                with open(train_path) as f_train, open(val_path) as f_val, open(
                    test_path
                ) as f_test:
                    train_results = [json.loads(result_line) for result_line in f_train]
                    val_results = [json.loads(result_line) for result_line in f_val]
                    test_results = [json.loads(result_line) for result_line in f_test]

                    train_df = pd.DataFrame(train_results)
                    val_df = pd.DataFrame(val_results)
                    test_df = pd.DataFrame(test_results)

                    value_combination_to_dfs[value_combination].append(
                        (train_df, val_df, test_df)
                    )
            except Exception as e:
                failed_value_combinations.append((value_combination, e))

    return value_combination_to_dfs, failed_value_combinations


def plot_nar_performance(
    glstm_nar_df: pd.DataFrame,
    other_narr_df: pd.DataFrame,
    metric_string: str,
    other_models: list[str],
    memory_dims: list[int],
    hidden_dims: list[int],
    axes: plt.Axes,
):
    for memory_dim in memory_dims:
        memory_dim_df = glstm_nar_df[(glstm_nar_df["memory_dim"] == memory_dim)]
        axes = plot_nar_result_from_df(
            memory_dim_df,
            axes,
            metric_string,
            plot_stderr=False,
            label=f"{MODEL_NAME} dim. {memory_dim}",
            marker="o",
            **PLOT_KWARGS,
        )

    # Initially just GCN for the paper
    for model in other_models:
        for hidden_dim in hidden_dims:
            hidden_dim_df = other_narr_df[
                (other_narr_df["model"] == model)
                & (other_narr_df["hidden_dim"] == hidden_dim)
            ]
            axes = plot_nar_result_from_df(
                hidden_dim_df,
                axes,
                metric_string,
                plot_stderr=False,
                label=f"{get_model_name(model)} dim. {hidden_dim}",
                marker="x",
                **PLOT_KWARGS,
            )

    axes.set_xlabel("Number of Neighbours")
    axes.set_ylabel("MSE" if metric_string.lower() == "mse" else metric_string.title())
    axes.grid(True)
    plt.tight_layout()

    return axes


def plot_trainable_params(
    glstm_nar_df, other_narr_df, other_models, memory_dims, hidden_dims, axes
):
    xlstm_memory_dim_to_trainable_params = glstm_nar_df.groupby("memory_dim")[
        "trainable_params"
    ].max()

    # Get trainable params for each model
    other_model_hidden_dim_to_trainable_params = {}
    for model in other_models:
        if len(other_narr_df[other_narr_df["model"] == model]) > 0:
            other_model_hidden_dim_to_trainable_params[model] = (
                other_narr_df[other_narr_df["model"] == model]
                .groupby("hidden_dim")["trainable_params"]
                .max()
            )

    # X positions for each model
    existing_models = [MODEL_NAME] + [
        m for m in other_models if m in other_model_hidden_dim_to_trainable_params
    ]
    x = np.arange(len(existing_models))
    bar_width = 0.2

    # Fetch values
    xlstm_vals = [xlstm_memory_dim_to_trainable_params.get(d, 0) for d in memory_dims]
    other_model_vals = []
    for model in other_models:
        if model in other_model_hidden_dim_to_trainable_params:
            model_vals = [
                other_model_hidden_dim_to_trainable_params[model].get(d, 0)
                for d in hidden_dims
            ]
            other_model_vals.append(model_vals)

    # Transpose for easier plotting
    model_groups = [xlstm_vals] + other_model_vals
    dim_groups = [memory_dims] + [hidden_dims] * len(other_model_vals)

    # Plot

    for i, (heights, dims) in enumerate(zip(model_groups, dim_groups)):
        for j, (height, dim) in enumerate(zip(heights, dims)):
            offset = (j - 1) * bar_width  # centers the 3 bars
            xpos = x[i] + offset
            bar = axes.bar(xpos, height, width=bar_width)
            axes.text(
                xpos,
                height + 0.02 * max(heights),
                f"{dim}",
                ha="center",
                va="bottom",
            )

    # Aesthetics
    axes.set_xticks(x)
    axes.set_xticklabels(
        [get_model_name(m) if m != MODEL_NAME else m for m in existing_models]
    )
    axes.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    axes.set_ylabel("Trainable Parameters")
    axes.margins(y=0.1)
    axes.set_yscale("log")

    return axes


def plot_nar_result_from_df(
    df: pd.DataFrame,
    axes: plt.Axes,
    metric_string: str,
    plot_stderr: bool = False,
    **kwargs,
):
    means = df.groupby("neighbour_count")[metric_string].mean()
    stds = df.groupby("neighbour_count")[metric_string].std()
    counts = df.groupby("neighbour_count").size()

    if plot_stderr:
        # Calculate standard error as std/sqrt(n)
        error_bars = stds / np.sqrt(counts)
    else:
        # Use standard deviation directly
        error_bars = stds

    # Plot mean line
    line = axes.plot(
        means.index,
        means.values,
        **kwargs,
    )[0]

    axes.fill_between(
        means.index,
        means.values - error_bars.values,
        means.values + error_bars.values,
        alpha=0.1,
        color=line.get_color(),
    )

    return axes


def derive_grad_metric_ratios(grad_df):
    possible_grad_metric_names = [
        "full_jacobian_norm",
        "key_jacobian_norm",
        "value_jacobian_norm",
        "jacobian_norm",
        "hessian_max_value",
        "hessian_mean_value",
        "hessian_frobenius_norm",
    ]
    # Filter to just those that appear in the df
    grad_metric_names = [
        grad_metric_name
        for grad_metric_name in possible_grad_metric_names
        if grad_metric_name in grad_df.columns
    ]
    # Due to a blunder while saving these, they don't have the graph index in the
    # dataframe, which I now want to compute ratios correctly with standard errors.
    # Therefore, we first go through and add it. We can work backwards to find the graph
    # index, since if a row has neighbour count N, there are N rows dedicated to each
    # graph.

    # Initialize output index list
    graph_indices = []
    ratio_values = {grad_metric_name: [] for grad_metric_name in grad_metric_names}
    i = 0  # current row index
    graph_id = 0  # graph index
    while i < len(grad_df):
        count = int(grad_df.loc[i, "neighbour_count"])
        graph_indices.extend([graph_id] * count)
        # Get the next #count rows
        next_rows = grad_df.iloc[i : i + count]
        # There should be exactly one is_selected_key here
        is_selected_key = next_rows["is_selected_key"]
        # Check that there is exactly one is_selected_key here
        assert is_selected_key.sum() == 1
        selected_row = next_rows[is_selected_key]

        # Now for each of the grad metric names, compute an associated ratio
        for grad_metric_name in grad_metric_names:
            selected_grad_metric_value = selected_row[grad_metric_name].values[0]
            ratios = selected_grad_metric_value / next_rows[grad_metric_name].values
            ratio_values[grad_metric_name].extend(list(ratios))

        i += count
        graph_id += 1
    grad_df["graph_index"] = graph_indices
    for grad_metric_name in grad_metric_names:
        grad_df[grad_metric_name + "_ratio"] = ratio_values[grad_metric_name]

    return grad_df


def load_grad_dicts(
    neighbour_counts,
    memory_dims,
    hidden_dims,
    k_hop_bools,
    result_dir,
    name_prefix,
    num_neighbours_name_component,
    pickle_name,
):
    glstm_grad_dicts = []
    gcn_grad_dicts = []

    grad_dict_neighbour_counts = [n for n in neighbour_counts if n <= 96]

    for k_hop_bool in k_hop_bools:
        for memory_dim in memory_dims:
            for neighbour_count in grad_dict_neighbour_counts:
                for seed in range(3):
                    file_path = os.path.join(
                        "results",
                        result_dir,
                        f"{name_prefix}-GNN-xLSTM-{num_neighbours_name_component}_{neighbour_count}_memory_dim_{memory_dim}{'_k-hop' if k_hop_bool else ''}",
                        str(seed),
                        f"{pickle_name}.pkl",
                    )
                    if not os.path.exists(file_path):
                        print(f"Missing file: {file_path}")
                        continue

                    with open(file_path, "rb") as f:
                        for grad_metrics in pkl.load(f):
                            # Convert dataclass into a dict
                            grad_metric_dict = asdict(grad_metrics)
                            grad_metric_dict["memory_dim"] = memory_dim
                            grad_metric_dict["neighbour_count"] = neighbour_count
                            grad_metric_dict["k_hop_bool"] = k_hop_bool
                            grad_metric_dict["seed"] = seed
                            glstm_grad_dicts.append(grad_metric_dict)

        for hidden_dim in hidden_dims:
            for neighbour_count in grad_dict_neighbour_counts:
                for seed in range(3):
                    file_path = os.path.join(
                        "results",
                        result_dir,
                        f"{name_prefix}-GCN-{num_neighbours_name_component}_{neighbour_count}_hidden_dim_{hidden_dim}{'_k-hop' if k_hop_bool else ''}",
                        str(seed),
                        f"{pickle_name}.pkl",
                    )
                    if not os.path.exists(file_path):
                        print(f"Missing file: {file_path}")
                        continue

                    with open(file_path, "rb") as f:
                        for grad_metrics in pkl.load(f):
                            grad_metric_dict = asdict(grad_metrics)
                            grad_metric_dict["hidden_dim"] = hidden_dim
                            grad_metric_dict["neighbour_count"] = neighbour_count
                            grad_metric_dict["k_hop_bool"] = k_hop_bool
                            grad_metric_dict["seed"] = seed
                            gcn_grad_dicts.append(grad_metric_dict)

    glstm_grad_df = pd.DataFrame(glstm_grad_dicts)
    gcn_grad_df = pd.DataFrame(gcn_grad_dicts)

    glstm_grad_df = derive_grad_metric_ratios(glstm_grad_df)
    gcn_grad_df = derive_grad_metric_ratios(gcn_grad_df)

    return glstm_grad_df, gcn_grad_df, grad_dict_neighbour_counts


def plot_combined_grad_metrics(
    glstm_grad_df,
    gcn_grad_df,
    memory_dims,
    hidden_dims,
    grad_metric_name,
    axes,
    plot_stderr: bool = False,
):
    for memory_dim in memory_dims:
        glstm_mask = glstm_grad_df["memory_dim"] == memory_dim

        glstm_grad_metric_means = (
            glstm_grad_df[glstm_mask]
            .groupby(["neighbour_count"])
            .mean()[grad_metric_name]
        )
        glstm_grad_metric_stds = (
            glstm_grad_df[glstm_mask]
            .groupby(["neighbour_count"])
            .std()[grad_metric_name]
        )
        glstm_grad_metric_counts = (
            glstm_grad_df[glstm_mask]
            .groupby(["neighbour_count"])
            .count()[grad_metric_name]
        )
        if plot_stderr:
            glstm_grad_metric_error = glstm_grad_metric_stds / np.sqrt(
                glstm_grad_metric_counts
            )
        else:
            glstm_grad_metric_error = glstm_grad_metric_stds

        # Plot mean line
        line = axes.plot(
            glstm_grad_metric_means.index,
            glstm_grad_metric_means.values,
            label=f"{MODEL_NAME} dim. {memory_dim}",
            marker="o",
            **PLOT_KWARGS,
        )[0]

        axes.fill_between(
            glstm_grad_metric_means.index,
            glstm_grad_metric_means.values - glstm_grad_metric_error.values,
            glstm_grad_metric_means.values + glstm_grad_metric_error.values,
            alpha=0.1,
            color=line.get_color(),
        )

    for hidden_dim in hidden_dims:
        gcn_mask = gcn_grad_df["hidden_dim"] == hidden_dim

        gcn_grad_metric_means = (
            gcn_grad_df[gcn_mask].groupby(["neighbour_count"]).mean()[grad_metric_name]
        )
        gcn_grad_metric_stds = (
            gcn_grad_df[gcn_mask].groupby(["neighbour_count"]).std()[grad_metric_name]
        )
        gcn_grad_metric_counts = (
            gcn_grad_df[gcn_mask].groupby(["neighbour_count"]).count()[grad_metric_name]
        )
        if plot_stderr:
            gcn_grad_metric_error = gcn_grad_metric_stds / np.sqrt(
                gcn_grad_metric_counts
            )
        else:
            gcn_grad_metric_error = gcn_grad_metric_stds

        line = axes.plot(
            gcn_grad_metric_means.index,
            gcn_grad_metric_means.values,
            label=f"GCN dim. {hidden_dim}",
            marker="x",
            **PLOT_KWARGS,
        )[0]

        axes.fill_between(
            gcn_grad_metric_means.index,
            gcn_grad_metric_means.values - gcn_grad_metric_error.values,
            gcn_grad_metric_means.values + gcn_grad_metric_error.values,
            alpha=0.1,
            color=line.get_color(),
        )

    axes.set_xlabel("Number of Neighbours")
    axes.set_ylabel(grad_metric_name.replace("_", " ").title())

    plt.tight_layout()
    axes.grid(True)

    return axes


def plot_separated_grad_and_ratio(
    glstm_grad_df,
    gcn_grad_df,
    grad_dict_neighbour_counts,
    grad_metric_name,
    ax1,
    ax2,
    ax3,
    task,
    plot_stderr: bool = False,
):
    # Left plot - xLSTM Selected vs Background values
    x = np.arange(len(grad_dict_neighbour_counts))
    width = 0.35

    memory_dim = 8 if task == "deterministic" else 16

    def _plot_separated(grad_df, ax, grad_metric_name, plot_stderr=False):
        selected_key_mask = grad_df["is_selected_key"]
        unselected_key_mask = ~grad_df["is_selected_key"]

        selected_key_grad_means = (
            grad_df[selected_key_mask]
            .groupby(["neighbour_count"])
            .mean()[grad_metric_name]
        )
        selected_key_grad_stds = (
            grad_df[selected_key_mask]
            .groupby(["neighbour_count"])
            .std()[grad_metric_name]
        )
        selected_key_grad_counts = (
            grad_df[selected_key_mask]
            .groupby(["neighbour_count"])
            .count()[grad_metric_name]
        )
        if plot_stderr:
            selected_key_grad_error = selected_key_grad_stds / np.sqrt(
                selected_key_grad_counts
            )
        else:
            selected_key_grad_error = selected_key_grad_stds
        unselected_key_grad_means = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .mean()[grad_metric_name]
        )
        unselected_key_grad_stds = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .std()[grad_metric_name]
        )
        unselected_key_grad_counts = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .count()[grad_metric_name]
        )
        if plot_stderr:
            unselected_key_grad_error = unselected_key_grad_stds / np.sqrt(
                unselected_key_grad_counts
            )
        else:
            unselected_key_grad_error = unselected_key_grad_stds

        # We ignore the selected keys here as the ratio is always taken w.r.t. them so
        # will always be 1.
        grad_ratio_means = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .mean()[grad_metric_name + "_ratio"]
        )
        grad_ratio_stds = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .std()[grad_metric_name + "_ratio"]
        )
        grad_ratio_counts = (
            grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .count()[grad_metric_name + "_ratio"]
        )
        if plot_stderr:
            grad_ratio_error = grad_ratio_stds / np.sqrt(grad_ratio_counts)
        else:
            grad_ratio_error = grad_ratio_stds

        ax.bar(
            x - width / 2,
            selected_key_grad_means,
            width,
            yerr=selected_key_grad_error,
            label="Selected",
        )
        ax.bar(
            x + width / 2,
            unselected_key_grad_means,
            width,
            yerr=unselected_key_grad_error,
            label="Background",
        )
        ax.set_xlabel("Number of Neighbours")
        ax.set_ylabel(grad_metric_name.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(grad_dict_neighbour_counts)
        ax.legend()

        return ax, grad_ratio_means, grad_ratio_error

    ax1, glstm_grad_ratios, glstm_grad_ratios_std = _plot_separated(
        glstm_grad_df[glstm_grad_df["memory_dim"] == memory_dim],
        ax1,
        grad_metric_name,
        plot_stderr,
    )
    ax1.set_title(f"{MODEL_NAME} dim. {memory_dim}")

    ax2, gcn_grad_ratios, gcn_grad_ratios_std = _plot_separated(
        gcn_grad_df[gcn_grad_df["hidden_dim"] == 64], ax2, grad_metric_name, plot_stderr
    )
    ax2.set_title(f"GCN dim. 64")

    # Plot gLSTM line with shading
    line = ax3.plot(x, glstm_grad_ratios, marker="o", label=MODEL_NAME, **PLOT_KWARGS)[
        0
    ]
    ax3.fill_between(
        x,
        glstm_grad_ratios - glstm_grad_ratios_std,
        glstm_grad_ratios + glstm_grad_ratios_std,
        alpha=0.1,
        color=line.get_color(),
    )

    # Plot GCN line with shading
    line = ax3.plot(x, gcn_grad_ratios, marker="o", label="GCN", **PLOT_KWARGS)[0]
    ax3.fill_between(
        x,
        gcn_grad_ratios - gcn_grad_ratios_std,
        gcn_grad_ratios + gcn_grad_ratios_std,
        alpha=0.1,
        color=line.get_color(),
    )
    ax3.set_xlabel("Number of Neighbours")
    ax3.set_ylabel(grad_metric_name.replace("_", " ").title() + " Ratio")
    ax3.set_title("Ratios")
    ax3.set_xticks(x)
    ax3.set_xticklabels(grad_dict_neighbour_counts)
    ax3.legend()
    plt.tight_layout()

    return ax1, ax2, ax3


def plot_jacobian_norm_ratios(
    glstm_grad_df,
    gcn_grad_df,
    memory_dims,
    hidden_dims,
    jacobian_norm_ratio_name,
    axes,
    plot_stderr: bool = False,
):
    for memory_dim in memory_dims:
        # We ignore the selected keys here as the ratio is always taken w.r.t. them so
        # will always be 1.
        unselected_key_mask = (~glstm_grad_df["is_selected_key"]) & (
            glstm_grad_df["memory_dim"] == memory_dim
        )
        glstm_jac_norm_ratio_means = (
            glstm_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .mean()[jacobian_norm_ratio_name]
        )
        glstm_jac_norm_ratio_stds = (
            glstm_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .std()[jacobian_norm_ratio_name]
        )
        glstm_jac_norm_ratio_counts = (
            glstm_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .count()[jacobian_norm_ratio_name]
        )
        if plot_stderr:
            glstm_jac_norm_ratio_error = glstm_jac_norm_ratio_stds / np.sqrt(
                glstm_jac_norm_ratio_counts
            )
        else:
            glstm_jac_norm_ratio_error = glstm_jac_norm_ratio_stds

        # Plot mean line
        line = axes.plot(
            glstm_jac_norm_ratio_means.index,
            glstm_jac_norm_ratio_means.values,
            label=f"{MODEL_NAME} dim. {memory_dim}",
            marker="o",
            **PLOT_KWARGS,
        )[0]
        axes.fill_between(
            glstm_jac_norm_ratio_means.index,
            glstm_jac_norm_ratio_means.values - glstm_jac_norm_ratio_error.values,
            glstm_jac_norm_ratio_means.values + glstm_jac_norm_ratio_error.values,
            alpha=0.1,
            color=line.get_color(),
        )

    model = "GCN"
    for hidden_dim in hidden_dims:
        unselected_key_mask = (~gcn_grad_df["is_selected_key"]) & (
            gcn_grad_df["hidden_dim"] == hidden_dim
        )
        gcn_jac_norm_ratio_means = (
            gcn_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .mean()[jacobian_norm_ratio_name]
        )
        gcn_jac_norm_ratio_stds = (
            gcn_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .std()[jacobian_norm_ratio_name]
        )
        gcn_jac_norm_ratio_counts = (
            gcn_grad_df[unselected_key_mask]
            .groupby(["neighbour_count"])
            .count()[jacobian_norm_ratio_name]
        )
        if plot_stderr:
            gcn_jac_norm_ratio_error = gcn_jac_norm_ratio_stds / np.sqrt(
                gcn_jac_norm_ratio_counts
            )
        else:
            gcn_jac_norm_ratio_error = gcn_jac_norm_ratio_stds

        line = axes.plot(
            gcn_jac_norm_ratio_means.index,
            gcn_jac_norm_ratio_means.values,
            label=f"GCN dim. {hidden_dim}",
            marker="x",
            **PLOT_KWARGS,
        )[0]
        axes.fill_between(
            gcn_jac_norm_ratio_means.index,
            gcn_jac_norm_ratio_means.values - gcn_jac_norm_ratio_error.values,
            gcn_jac_norm_ratio_means.values + gcn_jac_norm_ratio_error.values,
            alpha=0.1,
            color=line.get_color(),
        )

    axes.set_xlabel("Number of Neighbours")
    axes.set_ylabel("Jacobian Norm Ratios")

    axes.grid(True)

    plt.tight_layout()

    return axes


def plot_all_nar_results():
    neighbour_counts = [4, 8, 16, 32, 48, 64, 80, 96]
    memory_dims = [8, 16, 32]
    hidden_dims = [64, 128, 256]
    k_hop_bools = [True, False]
    plt.style.use(os.path.join("figures", "matlab.mplstyle"))

    for task, metric_string, goal in [
        # ("deterministic", "accuracy", max),
        ("classification", "accuracy", max),
        ("regression", "mse", min),
    ]:
        print(f"Plotting {task} results")

        if task == "classification":
            other_models = ["GCN", "GAT", "GatedGCN"]
            result_dir = "key_recall"
            name_prefix = "key-recall"
            num_neighbours_name_component = "num_neighbours"
        elif task == "regression":
            other_models = ["GCN"]
            result_dir = "key_recall_regression"
            name_prefix = "key-recall-regression"
            num_neighbours_name_component = "num_neighbours"
        elif task == "deterministic":
            other_models = ["GCN", "GatedGCN", "GNN-xLSTM-no-gates"]
            result_dir = "key_recall"
            name_prefix = "key-recall"
            num_neighbours_name_component = "deterministic_num_neighbours"
        else:
            raise ValueError(f"Unknown task: {task}")

        glstm_nar_df, _ = results_dirs_to_df(
            keywords_to_filepath=lambda neighbour_count, memory_dim, k_hop_bool: os.path.join(
                "results",
                result_dir,
                f"{name_prefix}-GNN-xLSTM-{num_neighbours_name_component}_{neighbour_count}_memory_dim_{memory_dim}{'_k-hop' if k_hop_bool else ''}",
            ),
            metric_name=metric_string,
            goal=lambda *_: goal,
            num_repeats=3,
            neighbour_count=neighbour_counts,
            memory_dim=memory_dims,
            k_hop_bool=k_hop_bools,
        )

        other_narr_df, _ = results_dirs_to_df(
            keywords_to_filepath=lambda model_name, neighbour_count, hidden_dim, k_hop_bool: os.path.join(
                "results",
                result_dir,
                f"{name_prefix}-{model_name}-{num_neighbours_name_component}_{neighbour_count}_hidden_dim_{hidden_dim}{'_k-hop' if k_hop_bool else ''}",
            ),
            metric_name=metric_string,
            goal=lambda *_: goal,
            num_repeats=3,
            model=other_models,
            neighbour_count=neighbour_counts,
            hidden_dim=hidden_dims,
            k_hop_bool=k_hop_bools,
        )

        # For the main paper - best GCN vs best gLSTM performance on NAR.
        print(f"NAR {task} results, showing gLSTM with K-hop vs others without.")
        _, axes = plt.subplots(1, 1, figsize=STANDARD_FIGSIZE)
        axes = plot_nar_performance(
            glstm_nar_df[glstm_nar_df["k_hop_bool"] == True],
            other_narr_df[other_narr_df["k_hop_bool"] == False],
            metric_string,
            ["GCN"],
            memory_dims,
            hidden_dims,
            axes,
        )
        MAKE_RIGHT_LEGEND(axes, 1)
        # Create directories if they don't exist
        save_path = os.path.join(
            "figures", "nar", task, "mixed_aggregation", "performance.pdf"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

        # For the appendix, separate by aggregation and plot all models.
        for k_hop_bool in k_hop_bools:
            print(
                f"NAR {task} results with extended model list, "
                + f"all with k-hop={k_hop_bool}."
            )
            _, axes = plt.subplots(1, 1, figsize=BIG_FIGSIZE)
            plot_nar_performance(
                glstm_nar_df[glstm_nar_df["k_hop_bool"] == k_hop_bool],
                other_narr_df[other_narr_df["k_hop_bool"] == k_hop_bool],
                metric_string,
                other_models,
                memory_dims,
                hidden_dims,
                axes,
            )
            MAKE_STANDARD_LEGEND(axes, 2)
            # Create directories if they don't exist
            save_path = os.path.join(
                "figures",
                "nar",
                task,
                "k_hop" if k_hop_bool else "no_k_hop",
                "expanded_performance.pdf",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

            plt.show()

        print(f"NAR {task} trainable parameters comparison - abridged.")
        _, axes = plt.subplots(figsize=PARAM_FIGSIZE)
        axes = plot_trainable_params(
            glstm_nar_df, other_narr_df, ["GCN"], memory_dims, hidden_dims, axes
        )
        save_path = os.path.join(
            "figures",
            "nar",
            task,
            "trainable_params.pdf",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

        print(f"NAR {task} trainable parameters comparison - expanded.")
        _, axes = plt.subplots(figsize=BIG_FIGSIZE)
        axes = plot_trainable_params(
            glstm_nar_df, other_narr_df, other_models, memory_dims, hidden_dims, axes
        )
        save_path = os.path.join(
            "figures",
            "nar",
            task,
            "trainable_params_expanded.pdf",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

        # Now, grad metrics
        glstm_grad_df, gcn_grad_df, grad_dict_neighbour_counts = load_grad_dicts(
            neighbour_counts,
            memory_dims,
            hidden_dims,
            k_hop_bools,
            result_dir,
            name_prefix,
            num_neighbours_name_component,
            "node_grads",
        )
        glstm_jacobian_df, gcn_jacobian_df, jacobian_dict_neighbour_counts = (
            load_grad_dicts(
                neighbour_counts,
                memory_dims,
                hidden_dims,
                k_hop_bools,
                result_dir,
                name_prefix,
                num_neighbours_name_component,
                "jacobian_metrics",
            )
        )

        # We do two things here: we compare k-hop gLSTM with non-k-hop GCN, to compare
        # the best performing model in each case. This is for the paper. We also compare
        # just k-hop models for more "apples to apples" comparison. This is for the
        # appendix.
        for gcn_k_hop_bool, aggregation_description in [
            (False, "mixed_aggregation"),
            (True, "k_hop"),
        ]:
            for vector_snippet in ["full", "key", "value"]:
                print(
                    f"NAR {task} combined Jacobian norms, {aggregation_description}, {vector_snippet}."
                )
                # Create a figure a single plot
                _, axes = plt.subplots(1, 1, figsize=HALF_FIGSIZE)
                axes = plot_combined_grad_metrics(
                    glstm_jacobian_df[glstm_jacobian_df["k_hop_bool"] == True],
                    gcn_jacobian_df[gcn_jacobian_df["k_hop_bool"] == gcn_k_hop_bool],
                    memory_dims,
                    hidden_dims,
                    f"{vector_snippet}_jacobian_norm",
                    axes,
                    plot_stderr=False,
                )
                # MAKE_STANDARD_LEGEND(axes, 2)
                save_path = os.path.join(
                    "figures",
                    "nar",
                    task,
                    aggregation_description,
                    f"{vector_snippet}_jacobian_norm_combined.pdf",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                axes.set_yscale("log")
                plt.savefig(save_path)
                plt.show()

                print(
                    f"NAR {task} separated Jacobian norms, {aggregation_description}, {vector_snippet}."
                )
                _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=BIG_FIGSIZE)
                ax1, ax2, ax3 = plot_separated_grad_and_ratio(
                    glstm_jacobian_df[glstm_jacobian_df["k_hop_bool"] == True],
                    gcn_jacobian_df[gcn_jacobian_df["k_hop_bool"] == gcn_k_hop_bool],
                    jacobian_dict_neighbour_counts,
                    f"{vector_snippet}_jacobian_norm",
                    ax1,
                    ax2,
                    ax3,
                    task,
                    plot_stderr=False,
                )
                plt.tight_layout()
                save_path = os.path.join(
                    "figures",
                    "nar",
                    task,
                    aggregation_description,
                    f"{vector_snippet}_jacobian_norm_separated.pdf",
                )
                ax3.set_yscale("log")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.show()

                print(
                    f"NAR {task} Jacobian norm ratios, {aggregation_description}, {vector_snippet}."
                )
                # Create a figure a single plot
                _, axes = plt.subplots(1, 1, figsize=HALF_FIGSIZE)
                axes = plot_jacobian_norm_ratios(
                    glstm_jacobian_df[glstm_jacobian_df["k_hop_bool"] == True],
                    gcn_jacobian_df[gcn_jacobian_df["k_hop_bool"] == gcn_k_hop_bool],
                    memory_dims,
                    hidden_dims,
                    f"{vector_snippet}_jacobian_norm_ratio",
                    axes,
                    plot_stderr=False,
                )
                MAKE_RIGHT_LEGEND(axes, 1)
                save_path = os.path.join(
                    "figures",
                    "nar",
                    task,
                    aggregation_description,
                    f"{vector_snippet}_jacobian_norm_ratios.pdf",
                )
                axes.set_yscale("log")
                axes.set_ylim(bottom=0.5)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.show()

            # Now, hessians
            print(
                f"NAR {task} separated Hessian max values, {aggregation_description}."
            )
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=BIG_FIGSIZE)
            ax1, ax2, ax3 = plot_separated_grad_and_ratio(
                glstm_grad_df[glstm_grad_df["k_hop_bool"] == True],
                gcn_grad_df[gcn_grad_df["k_hop_bool"] == gcn_k_hop_bool],
                grad_dict_neighbour_counts,
                "hessian_max_value",
                ax1,
                ax2,
                ax3,
                task,
                plot_stderr=False,
            )
            plt.tight_layout()
            save_path = os.path.join(
                "figures",
                "nar",
                task,
                aggregation_description,
                "separated_max_hessian_values.pdf",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.show()

            print(f"NAR {task} combined Hessian max values, {aggregation_description}.")
            _, axes = plt.subplots(1, 1, figsize=STANDARD_FIGSIZE)
            axes = plot_combined_grad_metrics(
                glstm_grad_df[glstm_grad_df["k_hop_bool"] == True],
                gcn_grad_df[gcn_grad_df["k_hop_bool"] == gcn_k_hop_bool],
                memory_dims,
                hidden_dims,
                "hessian_max_value",
                axes,
                plot_stderr=False,
            )
            axes.set_yscale("log")
            MAKE_RIGHT_LEGEND(axes, 1)
            save_path = os.path.join(
                "figures",
                "nar",
                task,
                aggregation_description,
                "max_hessian_values.pdf",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.show()

        # And finally, a plot where we check if different layer counts matter for GCN's
        # ability to solve the task
        layer_counts = [2, 3, 4, 5]
        gcn_variable_depth_df, _ = results_dirs_to_df(
            keywords_to_filepath=lambda neighbour_count, layer_count: os.path.join(
                "results",
                result_dir,
                f"{name_prefix}-GCN-{num_neighbours_name_component}_{neighbour_count}_hidden_dim_128{'' if layer_count == 2 else f'_{layer_count}_layers'}",
            ),
            metric_name=metric_string,
            goal=lambda *_: goal,
            num_repeats=3,
            neighbour_count=neighbour_counts,
            layer_count=layer_counts,
        )
        _, axes = plt.subplots(1, 1, figsize=STANDARD_FIGSIZE)
        for layer_count in layer_counts:
            layer_count_df = gcn_variable_depth_df[
                (gcn_variable_depth_df["layer_count"] == layer_count)
            ]
            axes = plot_nar_result_from_df(
                layer_count_df,
                axes,
                metric_string,
                plot_stderr=False,
                label=f"{layer_count} layers",
                marker="x",
                **PLOT_KWARGS,
            )

        axes.set_xlabel("Number of Neighbours")
        axes.set_ylabel(
            "MSE" if metric_string.lower() == "mse" else metric_string.title()
        )
        axes.grid(True)
        MAKE_STANDARD_LEGEND(axes, 3)
        plt.tight_layout()
        save_path = os.path.join(
            "figures",
            "nar",
            task,
            "no_k_hop",
            "gcn_variable_depth_performance.pdf",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


if __name__ == "__main__":
    plot_all_nar_results()
