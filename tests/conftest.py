import os

# Must be set before matplotlib is imported by intensity_fit.plotting.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from synth import SpeciesSpec, build_scenario  # noqa: E402


@pytest.fixture
def two_species(tmp_path):
    """Two noiseless species sharing one rotational temperature."""
    specs = [
        SpeciesSpec("alpha", base_freq=3000.0, n_a=5, n_c=0, scale=0.5, T=6.0),
        SpeciesSpec("beta", base_freq=4000.0, n_a=4, n_c=0, scale=2.0, T=6.0),
    ]
    return build_scenario(tmp_path, specs)
