import numpy as np
import pytest

from intensity_fit import pickett_meta as pm
from synth import SpeciesSpec, T_CAT, int_text, out_text


def test_existing_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert pm.existing_file(str(f)) == str(f)
    assert pm.existing_file(str(tmp_path / "missing")) is None
    assert pm.existing_file(None) is None
    assert pm.existing_file("") is None


def test_read_int_metadata_missing_file():
    meta = pm.read_int_metadata("/no/such/file.int")
    assert meta == {"exists": False, "dipoles": {}, "raw": []}
    assert pm.read_int_metadata(None)["exists"] is False


def test_read_int_metadata_parses_temperature_and_dipoles(tmp_path):
    path = tmp_path / "a.int"
    path.write_text(int_text(SpeciesSpec("a", 3000.0, dipoles=(1.5, 0.5, 2.5))))
    meta = pm.read_int_metadata(str(path))
    assert meta["exists"] is True
    assert meta["title"] == "a"
    assert meta["temperature_K"] == pytest.approx(T_CAT)
    assert meta["freq_max_GHz"] == pytest.approx(100.0)
    assert meta["dipoles"] == {"a": 1.5, "b": 0.5, "c": 2.5}


def test_read_int_metadata_dipole_component_from_id_without_comment(tmp_path):
    path = tmp_path / "b.int"
    path.write_text("t\n0 1 1.0 0 40 -10.0 -1.0 100.0 300.0 25\n 001 1.1\n 002 2.2\n 003 3.3\n")
    assert pm.read_int_metadata(str(path))["dipoles"] == {"a": 1.1, "b": 2.2, "c": 3.3}


def test_read_int_metadata_without_dipole_lines(tmp_path):
    path = tmp_path / "c.int"
    path.write_text("t\n0 1 1.0 0 40 -10.0 -1.0 100.0 77.0 25\n")
    meta = pm.read_int_metadata(str(path))
    assert meta["dipoles"] == {}
    assert meta["temperature_K"] == 77.0


def test_read_int_metadata_empty_file(tmp_path):
    path = tmp_path / "empty.int"
    path.write_text("")
    meta = pm.read_int_metadata(str(path))
    assert meta["exists"] is True
    assert "temperature_K" not in meta


def test_read_out_partition_function(tmp_path):
    path = tmp_path / "a.out"
    path.write_text(out_text())
    df = pm.read_out_partition_function(str(path))
    assert df["T_K"].tolist() == [2.0, 5.0, 10.0, 20.0, 50.0, 300.0]
    assert df["Q"].tolist() == pytest.approx([T ** 1.5 for T in df["T_K"]])


def test_read_out_partition_function_missing_or_without_table(tmp_path):
    assert pm.read_out_partition_function(None).empty
    assert list(pm.read_out_partition_function(None).columns) == ["T_K", "Q", "logQ"]
    path = tmp_path / "nt.out"
    path.write_text("nothing useful here\n")
    assert pm.read_out_partition_function(str(path)).empty


def _table(T, Q):
    import pandas as pd
    return pd.DataFrame({"T_K": T, "Q": Q})


def test_make_partition_function_exact_at_nodes_and_power_law_between():
    T = np.array([2.0, 5.0, 10.0, 300.0])
    q = pm.make_partition_function(_table(T, T ** 1.5))
    for t in T:
        assert q(t) == pytest.approx(t ** 1.5)
    # log-log interpolation reproduces a pure power law everywhere
    assert q(7.3) == pytest.approx(7.3 ** 1.5, rel=1e-9)


def test_make_partition_function_sorts_input():
    q = pm.make_partition_function(_table([10.0, 2.0, 5.0], [10.0 ** 1.5, 2.0 ** 1.5, 5.0 ** 1.5]))
    assert q(5.0) == pytest.approx(5.0 ** 1.5)


def test_make_partition_function_clamps_outside_table():
    T = np.array([2.0, 5.0, 10.0])
    q = pm.make_partition_function(_table(T, T ** 1.5))
    assert q(0.5) == pytest.approx(2.0 ** 1.5)
    assert q(1000.0) == pytest.approx(10.0 ** 1.5)


def test_make_partition_function_nonpositive_temperature_is_nan():
    q = pm.make_partition_function(_table([2.0, 5.0], [3.0, 11.0]))
    assert np.isnan(q(0.0))
    assert np.isnan(q(-4.0))


def test_make_partition_function_unusable_tables_give_none():
    assert pm.make_partition_function(_table([], [])) is None
    assert pm.make_partition_function(_table([5.0], [11.0])) is None
    assert pm.make_partition_function(_table([0.0, -1.0], [1.0, 2.0])) is None
    assert pm.make_partition_function(_table([2.0, 5.0], [0.0, -2.0])) is None


def test_read_egy(tmp_path):
    assert pm.read_egy(None).empty
    path = tmp_path / "a.egy"
    path.write_text("header\n1 2.5 3\n\nfoo\n4 5\n")
    df = pm.read_egy(str(path))
    assert df["line_number"].tolist() == [2, 5]
    assert df["values"].iloc[0] == [1.0, 2.5, 3.0]


def test_numbers_parses_scientific_and_signed():
    assert pm._numbers("a -1.5e-3 +2 .5 3.") == [-1.5e-3, 2.0, 0.5, 3.0]
    assert pm._numbers("none") == []


# ---------- dipoles are identified by ID (001/002/003), not by the comment ----------

HEADER = "t\n0 91 504.77 0 40 -9.0 -9.0 18 1.0\n"


def _int(tmp_path, body):
    path = tmp_path / "x.int"
    path.write_text(HEADER + body)
    return pm.read_int_metadata(str(path))


def test_wrong_comments_are_ignored(tmp_path):
    meta = _int(tmp_path, "   001    1.00     / a dipole moment\n   002    0.75     / a dipole moment\n   003    0.54     / c dipole moment\n")
    assert meta["dipoles"] == {"a": 1.00, "b": 0.75, "c": 0.54}
    assert meta["warnings"] == []


def test_comments_with_other_letters_are_ignored(tmp_path):
    meta = _int(tmp_path, "   001    2.54     / c dipole moment\n   002    1.20     / c dipole moment\n   003    0.15     / a dipole moment\n")
    assert meta["dipoles"] == {"a": 2.54, "b": 1.20, "c": 0.15}


def test_duplicate_id_is_added_like_spcat_and_flagged(tmp_path):
    meta = _int(tmp_path, "   001    2.65     / a dipole moment\n   001    0.30     / a dipole moment\n   003    0.20     / c dipole moment\n")
    assert meta["dipoles"]["a"] == pytest.approx(2.95)
    assert "b" not in meta["dipoles"]
    assert meta["dipole_ids"] == {"a": 2, "c": 1}
    assert len(meta["warnings"]) == 1
    assert "001" in meta["warnings"][0] and "2.95" in meta["warnings"][0]


def test_duplicate_values_keep_their_sign(tmp_path):
    meta = _int(tmp_path, "   001    2.65\n   001   -0.30\n")
    assert meta["dipoles"]["a"] == pytest.approx(2.35)


def test_non_standard_id_falls_back_to_the_comment(tmp_path):
    meta = _int(tmp_path, "   010    1.50     / b dipole moment\n")
    assert meta["dipoles"] == {"b": 1.5}


def test_non_standard_id_without_comment_is_ignored(tmp_path):
    assert _int(tmp_path, "   010    1.50\n")["dipoles"] == {}


def test_missing_ids_leave_components_undefined(tmp_path):
    meta = _int(tmp_path, "   001    0.43     / a dipole moment\n   003    1.21     / c dipole moment\n")
    assert meta["dipoles"] == {"a": 0.43, "c": 1.21}
