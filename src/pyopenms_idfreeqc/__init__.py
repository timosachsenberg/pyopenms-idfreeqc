"""Public package interface for the deterministic QC demo."""

from .calculate_metrics import (
    DEMO_DATASETS,
    METRIC_METADATA,
    ResolvedInput,
    build_qc_report,
    cli,
    compute_qc_metrics,
    main,
    plot_metrics_heatmap,
    prepare_demo_files,
    print_metrics_table,
    resolve_inputs,
    run_idfree_qc,
)

__all__ = [
    "DEMO_DATASETS",
    "METRIC_METADATA",
    "ResolvedInput",
    "build_qc_report",
    "cli",
    "compute_qc_metrics",
    "main",
    "plot_metrics_heatmap",
    "prepare_demo_files",
    "print_metrics_table",
    "resolve_inputs",
    "run_idfree_qc",
]

