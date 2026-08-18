"""Scatterplot and point-timeseries figures for EF RZSM evaluation."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from EF.evaluation import comparison_metrics


PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
}


def scatterplot(
    observed,
    predicted,
    pixel_count,
    data_type,
    area,
    output_file,
    axis_limits=(0.0, 0.55),
    bins=100,
    density_maximum=100,
):
    """Plot the pooled finite test-period EF pairs on a log-count density grid."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    paired = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[paired]
    predicted = predicted[paired]
    if len(observed) < 2:
        return None

    metrics = comparison_metrics(observed, predicted)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(10, 10), dpi=500)
        density = axis.hist2d(
            observed,
            predicted,
            bins=bins,
            cmap="jet",
            cmin=1,
            norm=LogNorm(vmin=1, vmax=density_maximum),
        )
        axis.plot(
            axis_limits,
            axis_limits,
            "k--",
            linewidth=1.5,
            label="1:1 Line",
        )
        colorbar = figure.colorbar(
            density[3],
            ax=axis,
            orientation="horizontal",
            shrink=0.5,
            pad=0.08,
        )
        colorbar.ax.tick_params(labelsize=14)
        colorbar.set_label("Count of Data Points (Log Scale)", fontsize=14)
        axis.set_xlabel(
            r"Observed RZSM (cm$^3$/cm$^3$)",
            fontsize=16,
            fontweight="bold",
        )
        axis.set_ylabel(
            r"Predicted RZSM (cm$^3$/cm$^3$)",
            fontsize=16,
            fontweight="bold",
        )
        axis.tick_params(labelsize=14)
        axis.set_xlim(*axis_limits)
        axis.set_ylim(*axis_limits)
        axis.set_title(
            f"Predicted RZSM Comparison ({data_type})",
            fontsize=18,
            fontweight="bold",
        )
        text = (
            f"Pixels: {pixel_count}\n"
            f"N: {metrics['N']}\n"
            f"R: {metrics['R']:.4f}\n"
            f"RMSE: {metrics['RMSE']:.4f}\n"
            f"Bias: {metrics['Bias']:.4f}\n"
            f"ubRMSE: {metrics['ubRMSE']:.4f}\n"
        )
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
        )
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_file)
        plt.close(figure)
    return metrics


def plot_timeseries(
    dates,
    surface_ssm,
    observed_rzsm,
    predicted_rzsm,
    metrics,
    latitude,
    longitude_west,
    output_file,
    figure_size,
    tick_dates,
    tick_format,
    x_limits=None,
):
    """Plot SSM, observed RZSM, and fixed-T EF RZSM for one point."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tick_dates = pd.to_datetime(tick_dates)
    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=figure_size)
        axis.plot(
            dates,
            surface_ssm,
            color="#b0b0b0",
            linewidth=1.5,
            label="SSM",
            alpha=0.7,
        )
        axis.plot(
            dates,
            observed_rzsm,
            color="#004488",
            linewidth=2,
            label="RZSM",
            alpha=0.8,
        )
        axis.plot(
            dates,
            predicted_rzsm,
            color="#CC3311",
            linewidth=3,
            label="RZSM Prediction",
        )
        if x_limits is not None:
            axis.set_xlim(pd.Timestamp(x_limits[0]), pd.Timestamp(x_limits[1]))
        axis.set_xticks(tick_dates)
        axis.set_xticklabels(
            [date.strftime(tick_format) for date in tick_dates],
            fontsize=14,
        )
        text = (
            f"R: {metrics['R']:.3f}\n"
            f"RMSE: {metrics['RMSE']:.3f}\n"
            f"Bias: {metrics['Bias']:.3f}\n"
            f"ubRMSE: {metrics['ubRMSE']:.3f}"
        )
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
            },
        )
        axis.set_xlabel("Time", fontsize=14)
        axis.tick_params(axis="y", labelsize=14)
        axis.set_ylabel(r"Soil Moisture (cm$^3$/cm$^3$)", fontsize=14)
        axis.set_ylim(0, 0.6)
        axis.set_title(
            f"RZSM Prediction Comparison ({latitude} N, {longitude_west} W)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        axis.legend(loc="upper right", fontsize=14, framealpha=0.8)
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_file, dpi=500, bbox_inches="tight")
        plt.close(figure)
    return output_file
