# Relative Intensity Fitting

Python tools for fitting relative rotational transition intensities from assigned
Pickett `.lin` files and predicted `.cat` files.

The workflow is intentionally close to `SPFIT`/`SPCAT` practice:

- run-specific inputs live in a config file;
- each run writes a compact human-readable `.ifit` report;
- detailed line and species diagnostics are saved as CSV;
- the exact config used for a run is copied into the output directory.

## Quick Start

```bash
cd /Users/filippobaroncelli/Desktop/coding/Python/relative_intensity
python -m intensity_fit run examples/fluorotelomer42_he.json
```

## Config Format

The config file is JSON. YAML is also accepted if `PyYAML` is installed.
See `examples/fluorotelomer42_he.json`.

Required top-level keys:

- `run`: run name, spectrum path, output directory.
- `fit`: fit options.
- `species`: list of species with `.lin`, `.cat`, and filters.

## Outputs

Each run writes:

- `<run_name>.ifit`: compact report for quick inspection.
- `species_summary.csv`: per-species summary.
- `line_diagnostics.csv`: per-line details and residuals.
- `observations.csv`: the observation table used in the optimizer.
- `config_used.json`: exact input config copied for reproducibility.
