import sys
import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import tqdm

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import statsmodels.api as sm
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

sys.path.append("./")
sys.path.append("src/")

from src import scaling_laws
# from src import madlad_constants  # Commented out as it doesn't exist yet


def num_fmt(n):
   for unit, div in zip(['', 'K', 'M', 'B', 'T'], [1, 1e3, 1e6, 1e9, 1e12]):
       if abs(n) < div * 1000:
           return f"{n/div:.{0 if abs(n) >= div * 10 else 1}f}{unit}"
   return f"{n/1e12:.1f}T"


def plot_colored_lines2(
    df, xlabel, ylabel,
    *,
    z_order=None,
    savepath=None,
    x_vlines=None, y_hlines=None,
    log_scale=None,
    colors=None,
    line_styles=None,           # ← flexible, user-driven
    default_line_style=None,    # ← optional global fallback
    key_points=None,
    figsize=(10, 6),
    title=None,
    annotate_points=True,
    legend_title='Category (z)'
):
    """
    Plot multiple lines coloured by `colors` and styled by `line_styles`.

    Parameters
    ----------
    line_styles : list | dict | None
        • dict  –  map *exact* z-labels to either a matplotlib linestyle string
                   ('--', ':', etc.) OR a dict of kwargs
        • list  –  same semantics as `colors`: ordered list matching `z_order`
        • None  –  all lines use `default_line_style`
    default_line_style : dict | str | None
        Global fallback style for any label not covered by `line_styles`.
        If None ➜ {'linestyle': '-', 'linewidth': 2.0, 'alpha': 1.0}
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)

    # ----- plotting order -----
    unique_labels = df['z'].unique()
    plot_order = (z_order or []) + [lbl for lbl in unique_labels if lbl not in (z_order or [])]

    # ----- colour mapping -----
    if colors is None:
        palette = sns.color_palette("viridis", len(plot_order))
        color_mapping = dict(zip(plot_order, palette))
    elif isinstance(colors, dict):
        color_mapping = colors
    else:                                    # list
        color_mapping = dict(zip(plot_order, colors))

    # ----- style mapping -------
    _fallback = {'linestyle': '-', 'linewidth': 2.0, 'alpha': 1.0}
    if isinstance(default_line_style, str):
        _fallback['linestyle'] = default_line_style
    elif isinstance(default_line_style, dict):
        _fallback.update(default_line_style)

    if line_styles is None:
        style_mapping = {lbl: _fallback for lbl in plot_order}

    elif isinstance(line_styles, dict):
        style_mapping = {
            lbl: (
                {'linestyle': line_styles[lbl]}           # <-- fix is here
                if isinstance(line_styles[lbl], str)
                else {**_fallback, **line_styles[lbl]}
            ) if lbl in line_styles else _fallback
            for lbl in plot_order
        }

    else:  # line_styles is a list
        style_mapping = {
            lbl: (
                {'linestyle': ls, **_fallback} if isinstance(ls, str) else {**_fallback, **ls}
            )
            for lbl, ls in zip(plot_order, line_styles)
        }

    # ----- main plot -----------
    for lbl in plot_order:
        subset = df[df['z'] == lbl]
        style = style_mapping[lbl]
        plt.plot(
            subset['x'], subset['y'],
            label=lbl,
            color=color_mapping[lbl],
            linestyle=style.get('linestyle', '-'),
            linewidth=style.get('linewidth', 2),
            alpha=style.get('alpha', 1.0),
            zorder=3
        )

    # ----- key points ----------
    if key_points:
        for lbl, pts in key_points.items():
            if lbl not in color_mapping:
                continue
            for x, y, marker in pts:
                plt.plot(x, y, marker, color=color_mapping[lbl],
                         markersize=8, alpha=0.7, zorder=4)
                if annotate_points:
                    plt.annotate(f'{x:.1f}x, {y:.1f}x',
                                 xy=(x, y), xytext=(5, 5),
                                 textcoords='offset points', fontsize=9, alpha=0.8)

    # ----- reference lines -----
    for x in x_vlines or []:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=1, zorder=0)
    for y in y_hlines or []:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=1, zorder=0)

    # ----- scales / labels -----
    if log_scale in {'x', 'both'}:
        plt.xscale('log')
    if log_scale in {'y', 'both'}:
        plt.yscale('log')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title or f'{ylabel} vs. {xlabel}', fontsize=16, fontweight='bold')
    plt.legend(title=legend_title, fontsize=12, title_fontsize='13', frameon=True)
    sns.despine(trim=True)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
    fig = plt.gcf()
    plt.close()
    return fig


def plot_optimal_scaling_line(
    scaling_laws: dict,
    compute_budget: float = None,
    target_loss: float = None,
    ax: plt.Axes = None,
    **kwargs
) -> plt.Axes:
    """
    Plot optimal scaling lines for different scaling laws.
    
    Parameters:
    -----------
    scaling_laws : dict
        Dictionary of scaling law names to ScalingLaw objects
    compute_budget : float, optional
        Compute budget in FLOPs
    target_loss : float, optional
        Target loss value
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure
    
    Returns:
    --------
    ax : matplotlib Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot logic will be implemented based on the specific requirements
    # This is a placeholder for now
    
    ax.set_xlabel("Training Tokens (D)")
    ax.set_ylabel("Model Size (N)")
    ax.set_title("Optimal Scaling Lines")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_tokens_per_param_ratio(
    scaling_laws: dict,
    compute_budgets: np.ndarray = None,
    ax: plt.Axes = None,
    **kwargs
) -> plt.Axes:
    """
    Plot the tokens per parameter ratio (D/N) vs compute budget.
    
    Parameters:
    -----------
    scaling_laws : dict
        Dictionary of scaling law names to ScalingLaw objects
    compute_budgets : np.ndarray, optional
        Array of compute budgets to evaluate
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure
    
    Returns:
    --------
    ax : matplotlib Axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if compute_budgets is None:
        compute_budgets = np.logspace(15, 25, 100)
    
    # Plot logic will be implemented based on the specific requirements
    # This is a placeholder for now
    
    ax.set_xlabel("Compute Budget (FLOPs)")
    ax.set_ylabel("Tokens per Parameter Ratio (D/N)")
    ax.set_title("Optimal Tokens per Parameter Ratio")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    return ax