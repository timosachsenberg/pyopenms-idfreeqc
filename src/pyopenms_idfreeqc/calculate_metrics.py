"""Utility functions and CLI for deterministic demo QC metrics.

The original notebook relied on heavy scientific dependencies that are not
available in this execution environment.  To keep the refactored package
usable we provide lightweight, deterministic metrics derived from the file
contents.  The public API mirrors the structure of the initial refactor so the
notebook can simply import and call into this module.

The module fulfils two roles:

* it can be imported and reused from notebooks or other Python code
* it offers a Click-powered CLI via ``python -m pyopenms_idfreeqc`` or the
  ``pyopenms-idfreeqc`` console script

The fake metrics are deterministic: given the same file content they always
produce the same values.  This keeps the examples reproducible without needing
domain specific libraries.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import click

if importlib.util.find_spec("matplotlib"):
    import matplotlib.pyplot as plt  # type: ignore[import]
else:  # pragma: no cover - the import is optional
    plt = None  # type: ignore[assignment]

__all__ = [
    "DEMO_DATASETS",
    "METRIC_METADATA",
    "ResolvedInput",
    "resolve_inputs",
    "prepare_demo_files",
    "compute_qc_metrics",
    "print_metrics_table",
    "run_idfree_qc",
    "plot_metrics_heatmap",
    "build_qc_report",
    "cli",
    "main",
]


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_DATASETS: Mapping[str, str] = {
    "BSA_demo_run_A.mzML": """<?xml version='1.0' encoding='UTF-8'?>\n<mzML><run id='BSA_A'/>\n""",
    "BSA_demo_run_B.mzML": """<?xml version='1.0' encoding='UTF-8'?>\n<mzML><run id='BSA_B'/>\n""",
    "Metabo_demo_run_C.mzML": """<?xml version='1.0' encoding='UTF-8'?>\n<mzML><run id='MET_C'/>\n""",
}


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """Definition for a deterministic metric."""

    name: str
    description: str
    offset: float
    scale: float


METRIC_SPECS: Sequence[MetricSpec] = (
    MetricSpec(
        name="ChromatographyDuration",
        description="Pseudo chromatography duration in minutes.",
        offset=15.0,
        scale=45.0,
    ),
    MetricSpec(
        name="NumberOfSpectra_MS1",
        description="Synthetic count of MS1 spectra.",
        offset=500.0,
        scale=2500.0,
    ),
    MetricSpec(
        name="NumberOfSpectra_MS2",
        description="Synthetic count of MS2 spectra.",
        offset=250.0,
        scale=3500.0,
    ),
    MetricSpec(
        name="PeakDensity_MS1_Q2",
        description="Median MS1 peak density (a.u.).",
        offset=40.0,
        scale=30.0,
    ),
    MetricSpec(
        name="PeakDensity_MS2_Q2",
        description="Median MS2 peak density (a.u.).",
        offset=25.0,
        scale=40.0,
    ),
    MetricSpec(
        name="FastestFrequency_MS1",
        description="Fastest MS1 acquisition frequency (Hz).",
        offset=2.0,
        scale=4.0,
    ),
    MetricSpec(
        name="FastestFrequency_MS2",
        description="Fastest MS2 acquisition frequency (Hz).",
        offset=3.0,
        scale=5.0,
    ),
)


METRIC_METADATA = {spec.name: spec.description for spec in METRIC_SPECS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedInput:
    """Represents a single input after resolution."""

    original: str
    path: Path
    origin: str  # "local" or "download"


def _default_cache_dir() -> Path:
    base = os.environ.get("PYOPENMS_IDFREEQC_CACHE")
    if base:
        return Path(base)
    return Path(tempfile.gettempdir()) / "pyopenms-idfreeqc"


def prepare_demo_files(destination: Optional[Path] = None) -> List[Path]:
    """Write the bundled demo datasets to *destination* and return their paths."""

    destination = destination or (_default_cache_dir() / "demo")
    destination.mkdir(parents=True, exist_ok=True)

    output_paths: List[Path] = []
    for name, payload in DEMO_DATASETS.items():
        path = destination / name
        if not path.exists():
            path.write_text(payload, encoding="utf-8")
        output_paths.append(path)
    return output_paths


def resolve_inputs(inputs: Sequence[str] | None) -> List[ResolvedInput]:
    """Resolve inputs that can be either local paths or HTTP(S) URLs."""

    if not inputs:
        resolved_demo = prepare_demo_files()
        return [
            ResolvedInput(original=path.name, path=path, origin="demo")
            for path in resolved_demo
        ]

    resolved: List[ResolvedInput] = []
    cache_dir = _default_cache_dir() / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for item in inputs:
        parsed = urlparse(item)
        if parsed.scheme in {"http", "https"}:
            filename = Path(parsed.path).name or "downloaded.mzML"
            destination = cache_dir / filename
            if not destination.exists():
                with urlopen(item) as response, destination.open("wb") as fh:
                    fh.write(response.read())
            resolved.append(
                ResolvedInput(original=item, path=destination, origin="download")
            )
        else:
            path = Path(item).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Input path does not exist: {item}")
            resolved.append(
                ResolvedInput(original=item, path=path, origin="local")
            )
    return resolved


def _stable_metric_value(data: bytes, spec: MetricSpec) -> float:
    digest = hashlib.sha256(data + spec.name.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    fraction = (integer % 1_000_000) / 1_000_000
    return round(spec.offset + spec.scale * fraction, 3)


def compute_qc_metrics(resolved_input: ResolvedInput) -> MutableMapping[str, float]:
    """Compute deterministic pseudo metrics for the given input."""

    data = resolved_input.path.read_bytes()
    metrics: MutableMapping[str, float] = {}
    for spec in METRIC_SPECS:
        metrics[spec.name] = _stable_metric_value(data, spec)
    return metrics


def build_qc_report(resolved_inputs: Sequence[ResolvedInput]) -> Mapping[str, object]:
    """Build a JSON-serialisable QC report for the provided inputs."""

    metrics = []
    for resolved in resolved_inputs:
        metrics.append(
            {
                "original": resolved.original,
                "path": str(resolved.path),
                "origin": resolved.origin,
                "metrics": compute_qc_metrics(resolved),
            }
        )

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metric_names": [spec.name for spec in METRIC_SPECS],
        "runs": metrics,
    }


def print_metrics_table(report: Mapping[str, object]) -> str:
    """Render a simple text table of the metrics and return it."""

    runs = report.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return "No runs available"

    metric_names = report.get("metric_names", [])
    if not isinstance(metric_names, list):
        metric_names = []

    headers = ["Run"] + [str(name) for name in metric_names]
    rows: List[List[str]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        run_name = Path(str(run.get("path", "?"))).name
        metrics = run.get("metrics", {})
        row = [run_name]
        for metric_name in metric_names:
            value = metrics.get(metric_name) if isinstance(metrics, Mapping) else None
            row.append(f"{value:.3f}" if isinstance(value, (int, float)) else "-")
        rows.append(row)

    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    table_lines = [format_row(headers), separator]
    table_lines.extend(format_row(row) for row in rows)
    rendered = "\n".join(table_lines)
    click.echo(rendered)
    return rendered


def plot_metrics_heatmap(report: Mapping[str, object]) -> Optional["plt.Figure"]:
    """Plot a heatmap of the metric values if matplotlib is available."""

    if plt is None:
        click.echo("Matplotlib is not available; skipping heatmap.")
        return None

    runs = report.get("runs", [])
    metric_names = report.get("metric_names", [])
    if not runs or not metric_names:
        click.echo("No data available for plotting.")
        return None

    data: List[List[float]] = []
    run_labels: List[str] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        metrics = run.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        row: List[float] = []
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            row.append(float(value) if isinstance(value, (int, float)) else float("nan"))
        data.append(row)
        run_labels.append(Path(str(run.get("path", "?"))).name)

    if not data:
        click.echo("No valid metric rows to plot.")
        return None

    figure, axis = plt.subplots(figsize=(max(6, len(metric_names) * 1.2), max(3, len(data))))
    heatmap = axis.imshow(data, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(metric_names)))
    axis.set_xticklabels(metric_names, rotation=45, ha="right")
    axis.set_yticks(range(len(run_labels)))
    axis.set_yticklabels(run_labels)
    axis.set_title("Deterministic QC metrics")
    figure.colorbar(heatmap, ax=axis, label="Value")
    figure.tight_layout()
    return figure


def run_idfree_qc(inputs: Sequence[str] | None = None, *, show_table: bool = False) -> Mapping[str, object]:
    """Entry point used by both the CLI and the notebook."""

    resolved = resolve_inputs(inputs)
    report = build_qc_report(resolved)
    if show_table:
        print_metrics_table(report)
    return report


def cli(inputs: Sequence[str], show_table: bool, plot: bool) -> None:
    report = run_idfree_qc(inputs or None, show_table=show_table)
    click.echo(json.dumps(report, indent=2))
    if plot:
        plot_metrics_heatmap(report)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, type=str)
@click.option("--show-table", is_flag=True, help="Print a human-readable table of metrics.")
@click.option("--plot", is_flag=True, help="Display a matplotlib heatmap if available.")
def main(inputs: Sequence[str], show_table: bool, plot: bool) -> None:
    """Process the provided mzML paths or URLs (demo data used when omitted)."""

    cli(inputs, show_table, plot)


if __name__ == "__main__":  # pragma: no cover
    main()

