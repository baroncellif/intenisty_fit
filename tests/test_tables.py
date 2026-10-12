import numpy as np
import pandas as pd
import pytest

from intensity_fit import tables
from intensity_fit.config import FitOptions, SpeciesConfig, load_run_config
from intensity_fit.constants import DEFAULT_CAT_TEMPERATURE_K
from intensity_fit.spectrum import read_spectrum
from synth import SpeciesSpec, T_CAT, build_scenario, cat_line, lin_line, transitions, true_intensity


def _species_cfg(scn, i=0, **kw):
    cfg = load_run_config(scn.config_path)
    sp = cfg.species[i]
    for k, v in kw.items():
        setattr(sp, k, v)
    return sp, cfg


@pytest.fixture
def mixed(tmp_path):
    return build_scenario(tmp_path, [SpeciesSpec("mix", 3000.0, n_a=4, n_c=3, dipoles=(2.0, 1.0, 0.5))])


def test_prediction_table_basic_columns_and_attrs(mixed):
    sp, cfg = _species_cfg(mixed)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert len(tab) == 7
    assert set(tab["species"]) == {"mix"}
    assert tab["I_cat"].tolist() == pytest.approx([10 ** t["lgint"] for t in transitions(mixed.specs[0])])
    assert tab["delta_MHz"].max() == pytest.approx(0.0)
    assert tab["cat_temperature_K"].eq(T_CAT).all()
    assert tab.attrs["cat_temperature_source"] == ".int"
    assert tab.attrs["dipoles"] == {"a": 2.0, "b": 1.0, "c": 0.5}
    assert tab.attrs["n_lin_read"] == tab.attrs["n_cat_read"] == 7
    assert tab.attrs["n_missing_qn"] == 0
    assert not tab.attrs["partition_table"].empty


def test_dipole_component_and_unit_dipole_intensity(mixed):
    sp, cfg = _species_cfg(mixed)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert tab["dipole_component"].tolist() == ["a"] * 4 + ["c"] * 3
    assert tab["dipole_used_D"].tolist() == pytest.approx([2.0] * 4 + [0.5] * 3)
    assert tab["I_cat_unit_dipole"].tolist() == pytest.approx((tab["I_cat"] / tab["dipole_used_D"] ** 2).tolist())


def test_missing_dipole_gives_nan_unit_intensity(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("nodip", 3000.0, n_a=3, with_int=False)])
    sp, cfg = _species_cfg(scn)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert tab["I_cat_unit_dipole"].isna().all()


def test_default_catalogue_temperature_without_int(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("noint", 3000.0, n_a=3, with_int=False)])
    sp, cfg = _species_cfg(scn)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert tab["cat_temperature_K"].eq(DEFAULT_CAT_TEMPERATURE_K).all()
    assert "default" in tab.attrs["cat_temperature_source"]


def test_partition_table_empty_without_out_file(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("noout", 3000.0, n_a=3, with_out=False)])
    sp, cfg = _species_cfg(scn)
    assert tables.build_species_prediction_table(sp, cfg.fit_options).attrs["partition_table"].empty


def test_filters_are_applied_to_lin_lines(mixed):
    sp, cfg = _species_cfg(mixed, filters={"freq_min": 3200.0})
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert (tab["lin_freq_MHz"] >= 3200.0).all()
    assert tab.attrs["lin_filtered_component_counts"] == {"a": 2, "c": 2}
    assert len(tab) == 4


def test_filters_removing_everything_raise(mixed):
    sp, cfg = _species_cfg(mixed, filters={"freq_min": 1e6})
    with pytest.raises(ValueError, match="after filtering"):
        tables.build_species_prediction_table(sp, cfg.fit_options)


def test_missing_cat_matches_are_dropped_and_counted(tmp_path):
    spec = SpeciesSpec("gap", 3000.0, n_a=4)
    scn = build_scenario(tmp_path, [spec])
    rows = transitions(spec)
    (tmp_path / "gap.cat").write_text("\n".join(cat_line(t, i + 1) for i, t in enumerate(rows) if i != 2) + "\n")
    sp, cfg = _species_cfg(scn)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert len(tab) == 3
    assert tab.attrs["n_missing_qn"] == 1
    assert tab.attrs["missing_qn_component_counts"] == {"a": 1}


def test_no_qn_match_at_all_raises(tmp_path):
    spec = SpeciesSpec("none", 3000.0, n_a=2)
    scn = build_scenario(tmp_path, [spec])
    other = [dict(t, qn=(9, 9, 9, 8, 8, 8)) for t in transitions(spec)[:1]]
    (tmp_path / "none.cat").write_text(cat_line(other[0]) + "\n")
    sp, cfg = _species_cfg(scn)
    with pytest.raises(ValueError, match="exact QN match"):
        tables.build_species_prediction_table(sp, cfg.fit_options)


def test_empty_lin_or_cat_raises(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("e", 3000.0, n_a=2)])
    sp, cfg = _species_cfg(scn)
    (tmp_path / "e.lin").write_text("")
    with pytest.raises(ValueError, match=".lin parser returned 0 rows"):
        tables.build_species_prediction_table(sp, cfg.fit_options)
    scn = build_scenario(tmp_path, [SpeciesSpec("e", 3000.0, n_a=2)])
    (tmp_path / "e.cat").write_text("")
    with pytest.raises(ValueError, match=".cat parser returned 0 rows"):
        tables.build_species_prediction_table(sp, cfg.fit_options)


def test_delta_max_filter(tmp_path):
    spec = SpeciesSpec("shift", 3000.0, n_a=3)
    scn = build_scenario(tmp_path, [spec])
    rows = transitions(spec)
    shifts = [0.001, 0.5, 0.002]
    (tmp_path / "shift.lin").write_text("\n".join(lin_line(t, t["freq"] + d) for t, d in zip(rows, shifts)) + "\n")
    sp, cfg = _species_cfg(scn, filters={"delta_max": 0.1})
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert len(tab) == 2
    assert tab["delta_MHz"].max() < 0.1


def test_delta_max_removing_everything_raises(tmp_path):
    spec = SpeciesSpec("shift", 3000.0, n_a=2)
    scn = build_scenario(tmp_path, [spec])
    rows = transitions(spec)
    (tmp_path / "shift.lin").write_text("\n".join(lin_line(t, t["freq"] + 1.0) for t in rows) + "\n")
    sp, cfg = _species_cfg(scn, filters={"delta_max": 0.1})
    with pytest.raises(ValueError, match="delta_max"):
        tables.build_species_prediction_table(sp, cfg.fit_options)


def test_species_attrs_pass_through(mixed):
    sp, cfg = _species_cfg(mixed, fixed_scale=3.0, fit_dipole_components=["A", "c"], delta_E_cm1=5.0, conformer_degeneracy=2.0)
    tab = tables.build_species_prediction_table(sp, cfg.fit_options)
    assert tab.attrs["fixed_scale"] == 3.0
    assert tab.attrs["fit_dipole_components"] == ["a", "c"]
    assert tab.attrs["delta_E_cm1"] == 5.0
    assert tab.attrs["conformer_degeneracy"] == 2.0


def test_build_species_tables_keys(two_species):
    cfg = load_run_config(two_species.config_path)
    assert list(tables.build_species_tables(cfg.species, cfg.fit_options)) == ["alpha", "beta"]


def test_observation_table_extracts_experimental_intensity(two_species):
    cfg = load_run_config(two_species.config_path)
    st = tables.build_species_tables(cfg.species, cfg.fit_options)
    obs = tables.build_observation_table(st, read_spectrum(cfg.spectrum_path), cfg.fit_options)
    assert len(obs) == 9
    expected = [true_intensity(t, s) for s in two_species.specs for t in transitions(s)]
    assert obs["exp_intensity"].tolist() == pytest.approx(expected, rel=1e-6)
    assert {"species", "I_cat", "elo_cm-1", "dr", "cat_temperature_K", "dipole_component"} <= set(obs.columns)


def test_observation_table_drops_lines_outside_spectrum(two_species):
    cfg = load_run_config(two_species.config_path)
    st = tables.build_species_tables(cfg.species, cfg.fit_options)
    spectrum = read_spectrum(cfg.spectrum_path)
    spectrum = spectrum[spectrum["freq_MHz"] < 3500.0]  # drops all of "beta"
    obs = tables.build_observation_table(st, spectrum, cfg.fit_options)
    assert set(obs["species"]) == {"alpha"}


def test_observation_table_without_any_intensity_raises(two_species):
    cfg = load_run_config(two_species.config_path)
    st = tables.build_species_tables(cfg.species, cfg.fit_options)
    far = pd.DataFrame({"freq_MHz": [1.0, 2.0], "intensity": [1.0, 1.0]})
    with pytest.raises(ValueError, match="No finite experimental intensities"):
        tables.build_observation_table(st, far, cfg.fit_options)


def test_residual_metrics():
    m = tables.residual_metrics([0.001, -0.003, np.nan])
    assert m["line_n_used"] == 2
    assert m["line_rmse"] == pytest.approx(np.sqrt((1e-6 + 9e-6) / 2) * 1e3)
    assert m["line_mae"] == pytest.approx(0.002)
    assert m["line_mean_residual"] == pytest.approx(-0.001)


def test_residual_metrics_empty():
    m = tables.residual_metrics([np.nan])
    assert m["line_n_used"] == 0
    assert np.isnan(m["line_rmse"])
