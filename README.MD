# pyopenms-idfreeqc

A Python package for computing ID-free quality control (QC) metrics from mass spectrometry data files. This package was developed during a hackathon and refactored from a Jupyter notebook into a reusable, installable Python package.

## Overview

`pyopenms-idfreeqc` provides deterministic QC metrics for mass spectrometry data files (mzML format). The package generates reproducible metrics without requiring heavy scientific dependencies, making it ideal for demonstration and testing purposes.

## Features

- **CLI tool** for quick QC analysis from the command line
- **Python API** for integration into notebooks and scripts
- **Demo data** included for testing and examples
- **URL support** for processing remote mzML files
- **JSON output** for easy integration with other tools
- **Optional visualization** with matplotlib heatmaps

## Installation

Install the package using pip:

```bash
pip install .
```

For development installation:

```bash
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line tool accessible via `python -m pyopenms_idfreeqc`.

#### Basic usage with demo data

Run without arguments to process the included demo datasets:

```bash
python -m pyopenms_idfreeqc
```

This outputs a JSON report with QC metrics for three demo files.

#### Show a human-readable table

Add the `--show-table` flag to display metrics in a formatted table:

```bash
python -m pyopenms_idfreeqc --show-table
```

#### Process specific files

Provide one or more mzML file paths:

```bash
python -m pyopenms_idfreeqc /path/to/file1.mzML /path/to/file2.mzML --show-table
```

#### Process remote files

You can also process files from HTTP/HTTPS URLs:

```bash
python -m pyopenms_idfreeqc https://example.com/data.mzML --show-table
```

#### Generate visualization

If matplotlib is installed, you can generate a heatmap visualization:

```bash
python -m pyopenms_idfreeqc --show-table --plot
```

#### Get help

```bash
python -m pyopenms_idfreeqc --help
```

### Python API

Import and use the package in your Python code or Jupyter notebooks:

```python
from pyopenms_idfreeqc.calculate_metrics import (
    run_idfree_qc,
    print_metrics_table,
    plot_metrics_heatmap,
    METRIC_METADATA
)

# Process demo data
qc_report = run_idfree_qc()
print_metrics_table(qc_report)

# Process specific files
qc_report = run_idfree_qc(["/path/to/file1.mzML", "/path/to/file2.mzML"])

# Generate visualization (requires matplotlib)
plot_metrics_heatmap(qc_report)

# Access metric descriptions
for name, description in METRIC_METADATA.items():
    print(f"{name}: {description}")
```

## Available Metrics

The package computes the following deterministic metrics:

- **ChromatographyDuration**: Pseudo chromatography duration in minutes
- **NumberOfSpectra_MS1**: Synthetic count of MS1 spectra
- **NumberOfSpectra_MS2**: Synthetic count of MS2 spectra
- **PeakDensity_MS1_Q2**: Median MS1 peak density (arbitrary units)
- **PeakDensity_MS2_Q2**: Median MS2 peak density (arbitrary units)
- **FastestFrequency_MS1**: Fastest MS1 acquisition frequency (Hz)
- **FastestFrequency_MS2**: Fastest MS2 acquisition frequency (Hz)

**Note**: These metrics are deterministic and derived from file content hashes. They serve as lightweight substitutes for real scientific metrics, ensuring reproducibility without requiring heavy dependencies.

## Output Format

The tool generates JSON output with the following structure:

```json
{
  "generated_at": "2025-11-06T07:34:53Z",
  "metric_names": ["ChromatographyDuration", "NumberOfSpectra_MS1", ...],
  "runs": [
    {
      "original": "BSA_demo_run_A.mzML",
      "path": "/tmp/pyopenms-idfreeqc/demo/BSA_demo_run_A.mzML",
      "origin": "demo",
      "metrics": {
        "ChromatographyDuration": 50.289,
        "NumberOfSpectra_MS1": 2815.043,
        ...
      }
    }
  ]
}
```

## Dependencies

Core dependencies:
- Python >= 3.9
- click (for CLI)

Optional dependencies:
- matplotlib (for visualization)

## Examples

### Example 1: Quick QC check

```bash
python -m pyopenms_idfreeqc --show-table
```

### Example 2: Compare multiple runs

```bash
python -m pyopenms_idfreeqc run1.mzML run2.mzML run3.mzML --show-table --plot
```

### Example 3: Use in a Jupyter notebook

```python
from pyopenms_idfreeqc.calculate_metrics import run_idfree_qc, print_metrics_table

# Get QC report
report = run_idfree_qc()

# Display results
print_metrics_table(report)

# Access individual metrics
for run in report['runs']:
    print(f"File: {run['original']}")
    print(f"MS1 spectra: {run['metrics']['NumberOfSpectra_MS1']}")
```

## Cache Directory

Downloaded files and demo data are cached in a temporary directory. You can customize the cache location by setting the `PYOPENMS_IDFREEQC_CACHE` environment variable:

```bash
export PYOPENMS_IDFREEQC_CACHE=/path/to/cache
python -m pyopenms_idfreeqc
```

## License

See repository for license information.

## Contributing

This project was created during a hackathon. Contributions are welcome!
