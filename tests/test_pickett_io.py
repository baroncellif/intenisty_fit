import pandas as pd
import pytest

from intensity_fit import pickett_io as pio
from synth import SpeciesSpec, cat_line, lin_line, transitions


# ---------- quantum-number decoding ----------

@pytest.mark.parametrize("text, expected", [
    ("", 0), ("  ", 0), (" 7", 7), ("12", 12), ("-3", -3),
    ("A0", 100), ("A5", 105), ("B2", 112), ("a1", -11), ("b2", -22),
])
def test_decode_qn_2char(text, expected):
    assert pio.decode_qn_2char(text) == expected


def test_decode_qn_2char_rejects_short_letter():
    with pytest.raises(ValueError):
        pio.decode_qn_2char("x")


def test_parse_qn_block_fixed_width():
    assert pio.parse_qn_block(" 1 0 1 0 0 0", n_total=6) == [1, 0, 1, 0, 0, 0]


def test_parse_qn_block_truncates_to_n_total():
    assert pio.parse_qn_block(" 1 2 3 4 5 6 7 8", n_total=4) == [1, 2, 3, 4]


def test_parse_qn_block_without_n_total_keeps_all():
    assert pio.parse_qn_block(" 1 2 3 4") == [1, 2, 3, 4]


def test_parse_qn_block_handles_letter_encoded_fields():
    assert pio.parse_qn_block("A1 2", n_total=2) == [101, 2]


@pytest.mark.parametrize("qnfmt, expected", [(303, 3), (404, 4), (1404, 4), (300, 10), (3, 3), (10, 10)])
def test_effective_nqn_from_qnfmt(qnfmt, expected):
    assert pio.effective_nqn_from_qnfmt(qnfmt) == expected


def test_pad_qn_to_12():
    assert pio.pad_qn_to_12([1, 2]) == [1, 2] + [0] * 10
    assert pio.pad_qn_to_12(list(range(15))) == list(range(12))
    assert len(pio.pad_qn_to_12([])) == 12


def test_qn_key_from_list_is_hashable_ints():
    key = pio.qn_key_from_list([1.0, 2, 3])
    assert key == (1, 2, 3)
    assert {key: 1}[key] == 1


def test_qn_columns_split_upper_lower():
    cols = pio._qn_columns([2, 1, 1, 1, 0, 1])
    assert (cols["Jp"], cols["Kap"], cols["Kcp"]) == (2, 1, 1)
    assert (cols["J"], cols["Ka"], cols["Kc"]) == (1, 0, 1)
    assert cols["nqn"] == 3
    assert len(cols["QN_all_padded"]) == 12


# ---------- .cat ----------

@pytest.fixture
def cat_file(tmp_path):
    rows = transitions(SpeciesSpec("x", 3000.0, n_a=3, n_c=1))
    path = tmp_path / "x.cat"
    path.write_text("\n".join(cat_line(t, i + 1) for i, t in enumerate(rows)) + "\n")
    return path, rows


def test_read_cat_roundtrip(cat_file):
    path, rows = cat_file
    df = pio.read_cat(str(path))
    assert len(df) == len(rows)
    assert df["freq_MHz"].tolist() == pytest.approx([t["freq"] for t in rows])
    assert df["lgint"].tolist() == pytest.approx([t["lgint"] for t in rows])
    assert df["elo_cm-1"].tolist() == pytest.approx([t["elo"] for t in rows])
    assert df["QN_all"].tolist() == [list(t["qn"]) for t in rows]
    assert set(df["nqn"]) == {3}
    assert df["tag"].tolist() == [1, 2, 3, 4]


def test_read_cat_skips_blank_and_malformed_lines(cat_file, tmp_path):
    path, rows = cat_file
    bad = tmp_path / "bad.cat"
    bad.write_text("\n" + path.read_text() + "this is not a catalogue line\n\n")
    assert len(pio.read_cat(str(bad))) == len(rows)


def test_read_cat_skips_lines_with_too_few_qn(tmp_path):
    t = transitions(SpeciesSpec("x", 3000.0, n_a=1))[0]
    truncated = cat_line(t)[:-4]  # only 4 of the 6 QN fields
    path = tmp_path / "short.cat"
    path.write_text(truncated + "\n")
    assert pio.read_cat(str(path)).empty


# ---------- .lin ----------

def test_read_lin_fixed_width(tmp_path):
    rows = transitions(SpeciesSpec("x", 3000.0, n_a=3, n_c=1))
    path = tmp_path / "x.lin"
    path.write_text("\n".join(lin_line(t) for t in rows) + "\n")
    df = pio.read_lin(str(path))
    assert len(df) == 4
    assert df["lin_freq_MHz"].tolist() == pytest.approx([t["freq"] for t in rows])
    assert df["lin_err_MHz"].iloc[0] == pytest.approx(0.01)
    assert df["wt"].iloc[0] == pytest.approx(1.0)
    assert df["QN_all"].tolist() == [list(t["qn"]) for t in rows]


def test_read_lin_tokenized_hand_edited(tmp_path):
    path = tmp_path / "hand.lin"
    path.write_text("1 0 1 0 0 0 3000.5 0.01 1.0\n")
    df = pio.read_lin(str(path))
    assert df["QN_all"].iloc[0] == [1, 0, 1, 0, 0, 0]
    assert df["lin_freq_MHz"].iloc[0] == pytest.approx(3000.5)


def test_read_lin_negative_frequency_uses_absolute_value(tmp_path):
    path = tmp_path / "neg.lin"
    path.write_text("1 0 1 0 0 0 -3000.5 -0.01 1.0\n")
    df = pio.read_lin(str(path))
    assert df["lin_freq_MHz"].iloc[0] == pytest.approx(3000.5)
    assert df["lin_err_MHz"].iloc[0] == pytest.approx(0.01)


def test_read_lin_drops_frequencies_above_90_ghz(tmp_path):
    path = tmp_path / "hi.lin"
    path.write_text("1 0 1 0 0 0 95000.0 0.01 1.0\n1 0 1 0 0 0 3000.0 0.01 1.0\n")
    assert pio.read_lin(str(path))["lin_freq_MHz"].tolist() == [3000.0]


def test_read_lin_drops_odd_qn_count_and_garbage(tmp_path):
    path = tmp_path / "odd.lin"
    path.write_text("1 0 1 0 0 3000.0 0.01 1.0\nnonsense\n\n")
    assert pio.read_lin(str(path)).empty


def test_parse_lin_line_default_weight():
    qn = "".join(f"{q:3d}" for q in (1, 0, 1, 0, 0, 0)).ljust(36)
    parsed = pio.parse_lin_line(f"{qn}3000.0 0.01")
    assert parsed is not None
    assert parsed[3] == 1.0


def test_parse_lin_line_returns_none_for_garbage():
    assert pio.parse_lin_line("abc") is None


# ---------- branches / components / filters ----------

@pytest.fixture
def transitions_df():
    # (Jp,Kap,Kcp,J,Ka,Kc): R-a, R-b, R-c, Q-a, P-a
    qn = [(2, 0, 2, 1, 0, 1), (2, 1, 2, 1, 0, 1), (2, 1, 1, 1, 0, 1), (1, 0, 1, 1, 0, 0), (0, 0, 0, 1, 0, 1)]
    df = pd.DataFrame(qn, columns=["Jp", "Kap", "Kcp", "J", "Ka", "Kc"])
    df["freq_MHz"] = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    df["elo_cm-1"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    return df


def test_classify_dipole_component(transitions_df):
    assert pio.classify_dipole_component(transitions_df).tolist() == ["a", "b", "c", "a", "a"]


def test_classify_dipole_component_handles_empty_frame():
    empty = pd.DataFrame(columns=["Jp", "Kap", "Kcp", "J", "Ka", "Kc"])
    assert pio.classify_dipole_component(empty).empty


def test_classify_unknown_when_both_parities_even():
    df = pd.DataFrame([(2, 0, 2, 1, 0, 2)], columns=["Jp", "Kap", "Kcp", "J", "Ka", "Kc"])
    assert pio.classify_dipole_component(df).tolist() == ["unknown"]


@pytest.mark.parametrize("branch, expected", [
    ("R", [True, True, True, False, False]),
    ("q", [False, False, False, True, False]),
    ("P", [False, False, False, False, True]),
    ("mu_a", [True, False, False, True, True]),
    ("mu_b", [False, True, False, False, False]),
    ("mu_c", [False, False, True, False, False]),
    ("all", [True] * 5),
    ("*", [True] * 5),
])
def test_infer_branch(transitions_df, branch, expected):
    assert pio.infer_branch(transitions_df, branch).tolist() == expected


def test_infer_branch_unknown_raises(transitions_df):
    with pytest.raises(ValueError):
        pio.infer_branch(transitions_df, "zzz")


def test_apply_filters_frequency_and_energy(transitions_df):
    out = pio.apply_filters(transitions_df, {"freq_min": 2000.0, "freq_max": 4000.0, "elo_max": 2.5})
    assert out["freq_MHz"].tolist() == [2000.0, 3000.0]
    assert out.index.tolist() == [0, 1]


def test_apply_filters_branch_combination(transitions_df):
    out = pio.apply_filters(transitions_df, {"branch_J": "R", "branch_mu": "mu_a"})
    assert out["freq_MHz"].tolist() == [1000.0]


def test_apply_filters_none_and_empty(transitions_df):
    assert len(pio.apply_filters(transitions_df, None)) == 5
    assert pio.apply_filters(transitions_df.iloc[0:0], {"freq_min": 1.0}).empty


def test_apply_filters_ignores_unknown_keys(transitions_df):
    assert len(pio.apply_filters(transitions_df, {"delta_max": 0.1})) == 5


# ---------- matching ----------

def _lin_cat(n_a=3, drop_cat=(), dup_lin=False):
    rows = transitions(SpeciesSpec("x", 3000.0, n_a=n_a))
    cat = pd.DataFrame([{
        "freq_MHz": t["freq"] + 0.02, "err_MHz": 0.001, "lgint": t["lgint"], "dr": 3,
        "elo_cm-1": t["elo"], "gup": 3, "tag": 1, "qnfmt": 303, "qn_raw": "",
        **pio._qn_columns(list(t["qn"])),
    } for i, t in enumerate(rows) if i not in drop_cat])
    lin_rows = rows + (rows[:1] if dup_lin else [])
    lin = pd.DataFrame([{**pio._qn_columns(list(t["qn"])), "lin_freq_MHz": t["freq"]} for t in lin_rows])
    return lin, cat


def test_match_lin_to_cat_exact_matches_by_qn():
    lin, cat = _lin_cat()
    merged = pio.match_lin_to_cat_exact(lin, cat)
    assert set(merged["match_mode"]) == {"QN"}
    assert (merged["cat_freq_MHz"] - merged["lin_freq_MHz"]).tolist() == pytest.approx([0.02] * 3)


def test_match_lin_to_cat_flags_missing_qn():
    lin, cat = _lin_cat(drop_cat=(1,))
    merged = pio.match_lin_to_cat_exact(lin, cat)
    assert merged["match_mode"].tolist() == ["QN", "MISSING_QN", "QN"]
    assert merged["cat_freq_MHz"].isna().tolist() == [False, True, False]


def test_match_lin_to_cat_duplicate_lin_raises():
    lin, cat = _lin_cat(dup_lin=True)
    with pytest.raises(ValueError, match="duplicated qn_key"):
        pio.match_lin_to_cat_exact(lin, cat)


def test_match_lin_to_cat_keeps_lowest_frequency_for_duplicate_cat():
    lin, cat = _lin_cat()
    extra = cat.iloc[[0]].copy()
    extra["freq_MHz"] += 5.0
    merged = pio.match_lin_to_cat_exact(lin, pd.concat([cat, extra], ignore_index=True))
    assert merged["cat_freq_MHz"].iloc[0] == pytest.approx(cat["freq_MHz"].iloc[0])
    assert len(merged) == 3


def test_check_duplicate_qn_ignores_empty():
    pio.check_duplicate_qn(pd.DataFrame(), "LIN", "lin_freq_MHz")
