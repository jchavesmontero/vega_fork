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
