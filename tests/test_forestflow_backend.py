"""Focused contracts for the optional external ForestFlow correction."""
import configparser
import numpy as np
import pytest

from vega.power_spectrum import PowerSpectrum


def _power(tracer1="LYA", tracer2="LYA", skip_peak=False):
    config = configparser.ConfigParser()
    config.read_dict({"model": {"bin_size_rp": "4", "bin_size_rt": "4",
                                 "small scale nl": "forestflow",
                                 "skip-nl-model-in-peak": str(skip_peak)}})
    fiducial = {"k": np.array([.1, .2, .4]), "pk_full": np.ones(3),
                "z_fiducial": 2.3, "z_eff": 2.3}
    return PowerSpectrum(config["model"], fiducial, {"name": tracer1, "type": "continuous"},
                         {"name": tracer2, "type": "continuous"}, "cf")


def test_forestflow_correction_auto_and_cross_conventions():
    correction = np.full((1000, 3), 4.0)
    assert np.all(_power().compute_dnl_forestflow({"forestflow_dnl": correction}) == 4)
    assert np.all(_power("LYA", "QSO").compute_dnl_forestflow({"forestflow_dnl": correction}) == 2)
    assert np.all(_power("QSO", "QSO").compute_dnl_forestflow({"forestflow_dnl": correction}) == 1)


def test_forestflow_correction_is_explicit_and_shaped():
    with pytest.raises(ValueError, match="requires"):
        _power().compute_dnl_forestflow({})
    with pytest.raises(ValueError, match="expected Vega"):
        _power().compute_dnl_forestflow({"forestflow_dnl": np.ones(3)})


def test_forestflow_correction_respects_skip_nl_model_in_peak():
    power = _power(skip_peak=True)
    correction = np.full(np.broadcast_shapes(power.muk_grid.shape,
                                              power.k_grid.shape), 2.0)
    params = {"bias_LYA": -.1, "beta_LYA": 1.0, "peak": True,
              "sigmaNL_par": 0., "sigmaNL_per": 0.,
              "forestflow_dnl": correction}
    peak = power.compute(np.ones(3), params)
    params["peak"] = False
    smooth = power.compute(np.ones(3), params)
    np.testing.assert_allclose(smooth, 2 * peak)


def _grid_ones(power):
    return np.ones(np.broadcast_shapes(power.muk_grid.shape, power.k_grid.shape))


def test_sii_x_sii_factor_and_decorrelation_act_only_on_sii_x_sii():
    """f exp(-(k_par sigma)^2 / 2) multiplies SiII autos and SiII x SiII crosses only."""
    params = {"forestflow_metal_sii_factor": 3.0, "forestflow_metal_sii_decorrelation": 2.0}
    for pair in (("SiII(1190)", "SiII(1190)"), ("SiII(1190)", "SiII(1193)"),
                 ("SiII(1193)", "SiII(1260)")):
        power = _power(*pair)
        factor = power.compute_dnl_forestflow({"forestflow_dnl": _grid_ones(power), **params})
        np.testing.assert_allclose(factor, 3.0 * np.exp(-0.5 * (power.k_par_grid * 2.0) ** 2))
    for pair in (("LYA", "SiII(1190)"), ("SiIII(1207)", "SiII(1190)"),
                 ("SiIII(1207)", "SiIII(1207)"), ("LYA", "LYA")):
        power = _power(*pair)
        assert np.all(power.compute_dnl_forestflow({"forestflow_dnl": _grid_ones(power), **params}) == 1)


def test_sii_x_sii_parameters_absent_leave_the_model_unchanged_and_are_validated():
    power = _power("SiII(1190)", "SiII(1193)")
    assert np.all(power.compute_dnl_forestflow({"forestflow_dnl": _grid_ones(power)}) == 1)
    for bad in ({"forestflow_metal_sii_factor": 0.0}, {"forestflow_metal_sii_factor": np.nan},
                {"forestflow_metal_sii_decorrelation": -1.0}):
        with pytest.raises(ValueError, match="forestflow_metal_sii"):
            power.compute_dnl_forestflow({"forestflow_dnl": _grid_ones(power), **bad})


def test_fast_metals_recomputes_sii_x_sii_only_when_its_parameters_change():
    """The frozen fast-metals metal x metal cache is refreshed for new SiII x SiII values."""
    from types import SimpleNamespace

    from vega.metals import Metals

    metals = Metals.__new__(Metals)
    metals.cache_xi_metal_metal = {}
    metals.Pk_metal = {("SiII(1190)", "SiII(1193)"): SimpleNamespace(_sii_x_sii=True),
                       ("SiIII(1207)", "SiII(1190)"): SimpleNamespace(_sii_x_sii=False)}
    calls = []

    def slow(pars, pk_lin, corr_hash, fast_metals):
        calls.append((corr_hash, pars.get("forestflow_metal_sii_factor")))
        return np.full(2, float(len(calls)))

    metals.compute_metal_corr_slow = slow
    sii, other = ("SiII(1190)", "SiII(1193)"), ("SiIII(1207)", "SiII(1190)")
    for factor in (1.0, 1.0, 2.0, 2.0):
        pars = {"forestflow_metal_sii_factor": factor}
        metals.compute_xi_metal_metal(None, pars, sii)
        metals.compute_xi_metal_metal(None, pars, other)
    assert calls == [(sii, 1.0), (other, 1.0), (sii, 2.0)]
    np.testing.assert_array_equal(metals.compute_xi_metal_metal(None, {"forestflow_metal_sii_factor": 2.0}, sii), 3.0)


def test_parameter_overrides_replace_main_config_parameters():
    from vega.vega_interface import apply_parameter_overrides

    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read_dict({"parameters": {"alpha_SiII(1190)": "1.", "bias_LYA": "-0.1"},
                      "sample": {"bias_LYA": "True"}})
    apply_parameter_overrides(config, None)
    assert config["parameters"]["alpha_SiII(1190)"] == "1."
    apply_parameter_overrides(config, {"alpha_SiII(1190)": 0.37, "alpha_SiIII(1207)": -0.2})
    assert config["parameters"].getfloat("alpha_SiII(1190)") == 0.37
    assert config["parameters"].getfloat("alpha_SiIII(1207)") == -0.2
    with pytest.raises(ValueError, match="sampled"):
        apply_parameter_overrides(config, {"bias_LYA": -0.2})
    with pytest.raises(ValueError, match="finite"):
        apply_parameter_overrides(config, {"alpha_SiII(1190)": np.nan})
