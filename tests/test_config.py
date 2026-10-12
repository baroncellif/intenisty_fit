import json

import pytest

from intensity_fit import config as cfgmod
from intensity_fit.config import FitOptions, load_run_config


def _write(tmp_path, data, name="c.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def minimal():
    return {
        "run": {"name": "r", "spectrum": "s.txt"},
        "species": [{"name": "A", "lin": "a.lin", "cat": "a.cat"}],
    }


def test_load_minimal_config_defaults(tmp_path, minimal):
    cfg = load_run_config(_write(tmp_path, minimal))
    assert cfg.name == "r"
    assert cfg.spectrum_path == "s.txt"
    assert cfg.output_dir == "r"  # defaults to the run name
    assert cfg.fit_options == FitOptions()
    sp = cfg.species[0]
    assert (sp.lin_file, sp.cat_file, sp.int_file) == ("a.lin", "a.cat", None)
    assert sp.conformer_degeneracy == 1.0
    assert cfg.raw == minimal


def test_prefix_resolves_all_companion_files(tmp_path, minimal):
    minimal["species"] = [{"name": "A", "prefix": "/d/a"}]
    sp = load_run_config(_write(tmp_path, minimal)).species[0]
    assert (sp.lin_file, sp.cat_file, sp.int_file, sp.str_file) == ("/d/a.lin", "/d/a.cat", "/d/a.int", "/d/a.str")
    assert (sp.egy_file, sp.out_file, sp.var_file) == ("/d/a.egy", "/d/a.out", "/d/a.var")


def test_explicit_paths_override_prefix(tmp_path, minimal):
    minimal["species"] = [{"name": "A", "prefix": "/d/a", "cat": "/other/x.cat", "int_file": "/other/x.int"}]
    sp = load_run_config(_write(tmp_path, minimal)).species[0]
    assert sp.cat_file == "/other/x.cat"
    assert sp.int_file == "/other/x.int"
    assert sp.lin_file == "/d/a.lin"


def test_species_options_are_read(tmp_path, minimal):
    minimal["species"][0].update({
        "filters": {"freq_min": 1.0}, "fixed_scale": 2.5, "fit_dipole_components": ["a"],
        "delta_E_cm-1": 12.0, "conformer_degeneracy": 2,
    })
    sp = load_run_config(_write(tmp_path, minimal)).species[0]
    assert sp.filters == {"freq_min": 1.0}
    assert sp.fixed_scale == 2.5
    assert sp.fit_dipole_components == ["a"]
    assert sp.delta_E_cm1 == 12.0
    assert sp.conformer_degeneracy == 2.0


def test_delta_e_alternative_key(tmp_path, minimal):
    minimal["species"][0]["delta_E_cm1"] = 7.0
    assert load_run_config(_write(tmp_path, minimal)).species[0].delta_E_cm1 == 7.0


def test_fit_options_are_passed_through(tmp_path, minimal):
    minimal["fit"] = {"fit_mode": "spectrum", "temp_init_K": 3.0, "lineshape": {"fwhm_MHz": 0.1}}
    fo = load_run_config(_write(tmp_path, minimal)).fit_options
    assert (fo.fit_mode, fo.temp_init_K, fo.lineshape) == ("spectrum", 3.0, {"fwhm_MHz": 0.1})


def test_unknown_fit_option_raises(tmp_path, minimal):
    minimal["fit"] = {"not_an_option": 1}
    with pytest.raises(TypeError):
        load_run_config(_write(tmp_path, minimal))


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["run"].pop("name"), "run.name"),
    (lambda c: c["run"].pop("spectrum"), "run.spectrum"),
    (lambda c: c.pop("species"), "at least one species"),
    (lambda c: c.__setitem__("species", []), "at least one species"),
    (lambda c: c["species"][0].pop("cat"), "lin and cat"),
    (lambda c: c["species"][0].pop("lin"), "lin and cat"),
])
def test_invalid_configs_raise(tmp_path, minimal, mutate, message):
    mutate(minimal)
    with pytest.raises(ValueError, match=message):
        load_run_config(_write(tmp_path, minimal))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "nope.json")


def test_jsonc_comments_and_trailing_commas(tmp_path):
    text = """
    {
      // line comment
      "run": {"name": "r", /* block */ "spectrum": "s.txt",},
      "species": [
        {"name": "A", "lin": "a.lin", "cat": "a.cat",},
      ],
    }
    """
    path = tmp_path / "c.jsonc"
    path.write_text(text)
    assert load_run_config(path).name == "r"


def test_jsonc_keeps_comment_markers_and_commas_inside_strings(tmp_path):
    text = '{"run": {"name": "a//b,]", "spectrum": "/* not a comment */ x.txt"}, "species": [{"name": "A", "lin": "l", "cat": "c"}]}'
    path = tmp_path / "c.jsonc"
    path.write_text(text)
    cfg = load_run_config(path)
    assert cfg.name == "a//b,]"
    assert cfg.spectrum_path == "/* not a comment */ x.txt"


def test_plain_json_with_comments_is_rejected(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"run": 1, // comment\n}')
    with pytest.raises(json.JSONDecodeError):
        load_run_config(path)


def test_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "c.yaml"
    path.write_text("run: {name: r, spectrum: s.txt}\nspecies:\n  - {name: A, lin: a.lin, cat: a.cat}\n")
    assert load_run_config(path).name == "r"


def test_yaml_without_pyyaml_gives_clear_error(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = tmp_path / "c.yml"
    path.write_text("run: {}")
    with pytest.raises(RuntimeError, match="PyYAML"):
        load_run_config(path)


def test_strip_helpers_directly():
    assert cfgmod._strip_jsonc_comments('{"a": 1 // x\n}').strip() == '{"a": 1 \n}'.strip()
    assert cfgmod._strip_jsonc_trailing_commas('{"a": [1, 2,],}') == '{"a": [1, 2]}'
    assert cfgmod._strip_jsonc_trailing_commas('{"a": "x,}"}') == '{"a": "x,}"}'
