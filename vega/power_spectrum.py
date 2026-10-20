import copy

import numpy as np
from numba import float64, njit

from . import utils


def _line_species(name):
    """Return the species of a metal tracer name, e.g. 'SiII' for 'SiII(1190)'."""
    return name.split("(")[0]


def sii_x_sii_parameters(params):
    """Return the validated SiII x SiII amplitude factor and decorrelation scale.

    Parameters
    ----------
    params : dict
        Computation parameters, optionally with ``forestflow_metal_sii_factor`` (``f > 0``)
        and ``forestflow_metal_sii_decorrelation`` (``sigma >= 0`` in Mpc/h).

    Returns
    -------
    tuple of float or None
        ``(f, sigma)``; None for a parameter that is not supplied.

    Raises
    ------
    ValueError
        If a supplied value is not finite, ``f <= 0`` or ``sigma < 0``.
    """
    factor = params.get("forestflow_metal_sii_factor")
    sigma = params.get("forestflow_metal_sii_decorrelation")
    if factor is not None:
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"forestflow_metal_sii_factor must be finite and > 0, got {factor}")
    if sigma is not None:
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(
                f"forestflow_metal_sii_decorrelation must be finite and >= 0 (Mpc/h), got {sigma}")
    return factor, sigma



class PowerSpectrum:
    """Power Spectrum computation and handling.

    # ! Slow operations should be kept in init as that is only called once

    # ! Compute is called many times and should be fast

    Extensions should have their separate method of the form
    'compute_extension' that can be called from outside
    """

    def __init__(self, config, fiducial, tracer1, tracer2, dataset_name=None, rmu_binning=False):
        """

        Parameters
        ----------
        config : dict
            pk config object
        fiducial : dict
            fiducial config
        tracer1 : dict
            Config of tracer 1
        tracer2 : dict
            Config of tracer 1
        dataset_name : string
            Name of dataset, by default None
        """
        self._config = config
        self.tracer1_name = copy.deepcopy(tracer1["name"])
        self.tracer2_name = copy.deepcopy(tracer2["name"])
        self._corr_name = f"{self.tracer1_name}x{self.tracer2_name}"
        self.tracer1_type = copy.deepcopy(tracer1["type"])
        self.tracer2_type = copy.deepcopy(tracer2["type"])

        self._name = dataset_name
        # SiII x SiII correlation (autos and crosses of the SiII lines, not SiIII), which can
        # receive the forestflow_metal_sii_* amplitude factor and decorrelation
        self._sii_x_sii = all(
            tracer["type"] == "continuous" and _line_species(tracer["name"]) == "SiII"
            for tracer in (tracer1, tracer2)
        )
        self.k_grid = fiducial["k"]
        self.rmu_binning = rmu_binning
        self.use_Gk = self._config.getboolean("model binning", True)

        self.skip_nl_model_in_peak = config.getboolean("skip-nl-model-in-peak", False)

        if self.rmu_binning:
            self._bin_size_r = config.getfloat("bin_size_r")
            # mu smoothing does not make sense
            # self._bin_size_mu = config.getfloat('bin_size_mu')
        else:
            self._bin_size_rp = config.getfloat("bin_size_rp")
            self._bin_size_rt = config.getfloat("bin_size_rt")

        # Damping scale for P(k) - used to match EFT behavior
        self.pk_damping_scale = config.getfloat("pk-damping-scale", None)
        self.pk_damping_power = config.getint("pk-damping-power", 2)

        # Get the HCD model and check for UV
        self.hcd_model = self._config.get("model-hcd", None)
        self._add_uvb = self._config.getboolean("UVB-fluctuations", False)
        self._add_heii = self._config.getboolean("HeII-reionization", False)

        # Check the HCD model
        self._Fvoigt_data = None
        if self.hcd_model is not None and "fvoigt" in self.hcd_model:
            assert "fvoigt_model" in self._config.keys(), "No fvoigt_model specified in config"

            fvoigt_model = self._config.get("fvoigt_model")
            if "/" not in fvoigt_model:
                path = utils.find_file(f"fvoigt_models/Fvoigt_{fvoigt_model}.txt")
            else:
                path = fvoigt_model
            self._Fvoigt_data = np.loadtxt(path)

        # Initialize some stuff we need
        self.pk_Gk = None
        self._pk_fid = (
            fiducial["pk_full"] * ((1 + fiducial["z_fiducial"]) / (1.0 + fiducial["z_eff"])) ** 2
        )

        # Initialize the mu_k grid
        num_bins_muk = self._config.getint("num_bins_muk", 1000)
        muk_grid = (np.arange(num_bins_muk) + 0.5) / num_bins_muk
        self.muk_grid = muk_grid[:, None]

        self.k_par_grid = self.k_grid * self.muk_grid
        self.k_trans_grid = self.k_grid * np.sqrt(1 - self.muk_grid**2)
        self._arinyo_pars = None
        self._peak_nl_pars = None
        self._L0_hcd_cache = None
        self._F_hcd_cache = None

    def compute(self, pk_lin, params, fast_metals=False):
        """Computes a power spectrum for the tracers using the input
        linear P(k) and parameters.

        Parameters
        ----------
        pk_lin : 1D array
            Linear Power Spectrum
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Power spectrum
        """
        # params = params

        # Get the core biases and betas
        bias_beta = utils.bias_beta(params, self.tracer1_name, self.tracer2_name)
        bias1, beta1, bias2, beta2 = bias_beta

        # Add UVB fluctuations and HeII reionization models
        if self._add_uvb or self._add_heii:
            if self.tracer1_name == "LYA":
                bias1, beta1 = self.compute_bias_beta_uv_heii(bias1, beta1, params)
            if self.tracer2_name == "LYA":
                bias2, beta2 = self.compute_bias_beta_uv_heii(bias2, beta2, params)

        # Add HCD model
        if self.hcd_model is not None:
            if self.tracer1_name == "LYA":
                bias1, beta1 = self.compute_bias_beta_hcd(bias1, beta1, params)
            if self.tracer2_name == "LYA":
                bias2, beta2 = self.compute_bias_beta_hcd(bias2, beta2, params)

        # Compute kaiser model
        pk_full = pk_lin * self.compute_kaiser(bias1, beta1, bias2, beta2, fast_metals)

        # add non linear small scales
        skip_nl = self.skip_nl_model_in_peak and params["peak"]
        if "small scale nl" in self._config.keys() and not skip_nl:
            nonlinear_mode = self._config.get("small scale nl").strip().lower()
            if nonlinear_mode == "forestflow":
                pk_full *= self.compute_dnl_forestflow(params)
            elif nonlinear_mode == "arinyo":
                pk_full *= self.compute_dnl_arinyo(params)
            elif nonlinear_mode == "mcdonald":
                pk_full *= self.compute_dnl_mcdonald()
            else:
                print("small scale nl: must be mcdonald, arinyo, or forestflow")
                raise ValueError("Incorrect 'small scale nl' specified")

        # model the effect of binning
        if self.use_Gk:
            if self.pk_Gk is None:
                self.pk_Gk = self.compute_Gk(params)
            pk_full *= self.pk_Gk

        if "mock-bin-size" in self._config:
            bin_size = self._config.getfloat("mock-bin-size")
            smoothing_parameters = {
                f"par binsize {self._name}": bin_size,
                f"per binsize {self._name}": bin_size,
            }

            los_smoothing = self._config.get("mock-los-smoothing")
            if los_smoothing == "growth":
                smoothing_parameters[f"par binsize {self._name}"] *= 1 + params["growth_rate"]
            elif los_smoothing == "amplitude":
                smoothing_parameters[f"par binsize {self._name}"] *= 1 + params["los_smooth_amp"]
            elif los_smoothing == "only-los":
                smoothing_parameters[f"per binsize {self._name}"] = 0
            elif los_smoothing is not None:
                raise ValueError(f"Unknown mock LOS smoothing option {los_smoothing}.")

            pk_full *= self.compute_Gk(smoothing_parameters)

        # add non linear large scales
        if params["peak"]:
            pk_full *= self.compute_peak_nl(params)

        # add full shape smoothing
        if "fullshape smoothing" in self._config and not skip_nl:
            smoothing_type = self._config.get("fullshape smoothing")
            if "gauss" in smoothing_type:
                pk_full *= self.compute_fullshape_gauss_smoothing(params)
            elif "exp" in smoothing_type:
                pk_full *= self.compute_fullshape_exp_smoothing(params)
            else:
                raise ValueError('"fullshape smoothing" must be of type "gauss" or "exp".')

        # add velocity dispersion
        if "velocity dispersion" in self._config:
            smoothing_type = self._config.get("velocity dispersion")
            if "gauss" in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_gauss(params)
            elif "lorentz" in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_lorentz(params)
            elif "lorentz_gauss" in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_lorentz(params)
                pk_full *= self.compute_velocity_dispersion_gauss(params)
            else:
                raise ValueError('"velocity dispersion" must be of type "gauss" or "lorentz".')

        # P(k) damping at high k
        if self.pk_damping_scale is not None:
            pk_full *= utils.compute_kn_smoothing(
                self.pk_damping_scale, self.k_grid.astype(float), n=self.pk_damping_power
            )

        return pk_full

    def compute_kaiser(self, bias1, beta1, bias2, beta2, fast_metals=False):
        """Compute Kaiser model.

        Parameters
        ----------
        bias1 : float
            Bias for tracer 1
        beta1 : float
            Beta for tracer 1
        bias2 : float
            Bias for tracer 2
        beta2 : float
            Beta for tracer 2

        Returns
        -------
        ND Array
            Kaiser term
        """
        pk = 1 + beta1 * self.muk_grid**2
        pk = pk * (1 + beta2 * self.muk_grid**2)

        if not fast_metals:
            pk *= bias1 * bias2
        return pk

    def compute_bias_beta_uv_heii(self, bias, beta, params):
        """Compute effective biases that include UV and helium reionization modeling.

        Parameters
        ----------
        bias : float
            Bias for tracer
        beta : float
            Beta for tracer
        params : dict
            Computation parameters

        Returns
        -------
        (float, float)
            Effective bias and beta
        """
        bias_eff = bias

        if self._add_uvb:
            bias_gamma = params["bias_gamma"]
            bias_prim = params["bias_prim"]
            lambda_uv = params["lambda_uv"]

            W = np.arctan(self.k_grid * lambda_uv) / (self.k_grid * lambda_uv)
            bias_eff += bias_gamma * W / (1 + bias_prim * W)

        if self._add_heii:
            bias_gamma_e = params["bias_gamma_e"]
            bias_prim = params["bias_prim"]
            lambda_heii = params["lambda_HeII"]

            W = np.arctan(self.k_grid * lambda_heii) / (self.k_grid * lambda_heii)
            bias_eff += bias_gamma_e * W / (1 + bias_prim * W)

        beta_eff = beta * bias / bias_eff

        return bias_eff, beta_eff

    def compute_bias_beta_hcd(self, bias, beta, params):
        """Compute effective biases that include HCD modeling.

        Parameters
        ----------
        bias : float
            Bias for tracer
        beta : float
            Beta for tracer
        params : dict
            Computation parameters

        Returns
        -------
        (float, float)
            Effective bias and beta
        """
        # Check if we have an HCD bias for each component
        bias_hcd = params.get(f"bias_hcd_{self._corr_name}", None)
        if bias_hcd is None:
            bias_hcd = params["bias_hcd"]

        # Get the other parameters
        beta_hcd = params.get(f"beta_hcd_{self._corr_name}", None)
        if beta_hcd is None:
            beta_hcd = params["beta_hcd"]

        # Check which model we need
        if "Rogers" in self.hcd_model:
            L0 = params["L0_hcd"]
            self._compute_hcd_cached(self._hcd_Rogers2018, L0, self.k_par_grid)
        elif "fvoigt" in self.hcd_model:
            assert self._Fvoigt_data is not None
            L0 = params.get("L0_fvoigt", 1)
            self._compute_hcd_cached(self._hcd_fvoigt, L0)
        elif "sinc" in self.hcd_model:
            L0 = params.get("L0_sinc", 1)
            self._compute_hcd_cached(self._hcd_sinc, L0)
        else:
            raise ValueError(
                f"Unknown hcd model {self.hcd_model}. Choose from ['Rogers', 'fvoigt', 'sinc']"
            )

        bias_eff = bias + bias_hcd * self._F_hcd
        beta_eff = bias * beta + bias_hcd * beta_hcd * self._F_hcd
        beta_eff /= bias + bias_hcd * self._F_hcd

        return bias_eff, beta_eff

    def _compute_hcd_cached(self, func, L0_hcd, *args):
        """Compute and cache the HCD model F(k) if L0_hcd has changed.

        Parameters
        ----------
        func : callable
            HCD model function to call
        L0_hcd : float
            Characteristic HCD length scale; used as the cache key
        *args : additional arguments passed to func
        """
        if L0_hcd != self._L0_hcd_cache or self._F_hcd is None:
            self._F_hcd = func(L0_hcd, *args)
            self._L0_hcd_cache = L0_hcd

    def _hcd_sinc(self, L0):
        """HCD sinc model.

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        return utils.sinc(self.k_par_grid * L0)

    @staticmethod
    @njit(float64[:, :](float64, float64[:, :]))
    def _hcd_Rogers2018(L0, k_par_grid):
        """Model the effect of HCD systems with the Fourier transform
        of a Lorentzian profile. Motivated by Rogers et al. (2018).

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        f_hcd = np.exp(-L0 * k_par_grid)
        return f_hcd

    def _hcd_fvoigt(self, L0):
        """Use Fvoigt function to fit the DLA in the autocorrelation Lyman-alpha
        without masking them ! (L0 = 1)

        (If you want to mask them use Fvoigt_exp.txt and L0 = 10 as eBOSS DR14)

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        k_data = self._Fvoigt_data[:, 0]
        F_data = self._Fvoigt_data[:, 1]
        F_hcd = np.interp(L0 * self.k_par_grid, k_data, F_data, left=1, right=0)

        return F_hcd

    def compute_peak_nl(self, params):
        """Compute the non-linear gaussian correction for the peak component.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor for the peak
        """
        sigma_par = params.get("sigmaNL_par", None)
        sigma_trans = params.get("sigmaNL_per", None)
        growth_rate = params.get("growth_rate")

        if sigma_par is None and sigma_trans is not None:
            sigma_par = sigma_trans * (1 + growth_rate)
        elif sigma_trans is None and sigma_par is not None:
            sigma_trans = sigma_par / (1 + growth_rate)
        elif sigma_par is None and sigma_trans is None:
            raise ValueError("No parameters for peak NL found. Add sigmaNL_par and/or sigmaNL_par.")

        if self._peak_nl_pars is None:
            self._peak_nl_pars = np.array([sigma_par, sigma_trans]) + 1

        if not np.allclose(np.array([sigma_par, sigma_trans]), self._peak_nl_pars):
            peak_nl = self.k_par_grid**2 * sigma_par**2
            peak_nl += self.k_trans_grid**2 * sigma_trans**2
            self._peak_nl_pars = np.array([sigma_par, sigma_trans])
            self._peak_nl_cache = np.exp(-peak_nl / 2)

        return self._peak_nl_cache

    def compute_dnl_mcdonald(self):
        """Non linear term from McDonald 2003.

        Returns
        -------
        ND Array
            D_NL factor
        """
        assert self.tracer1_name == "LYA"
        assert self.tracer2_name == "LYA"

        kvel = 1.22 * (1 + self.k_grid / 0.923) ** 0.451
        dnl = (self.k_grid / 6.4) ** 0.569 - (self.k_grid / 15.3) ** 2.01
        dnl = dnl - (self.k_grid * self.muk_grid / kvel) ** 1.5
        return np.exp(dnl)

    def compute_dnl_arinyo(self, params):
        """Non linear term from Arinyo et al 2015.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            D_NL factor
        """
        two_lya_flag = "LY" in self.tracer1_name and "LY" in self.tracer2_name
        one_lya_flag = "LY" in self.tracer1_name or "LY" in self.tracer2_name

        q1 = params["dnl_arinyo_q1"]
        kv = params["dnl_arinyo_kv"]
        av = params["dnl_arinyo_av"]
        bv = params["dnl_arinyo_bv"]
        kp = params["dnl_arinyo_kp"]

        q2 = params.get("dnl_arinyo_q2", 0)

        if self._arinyo_pars is None:
            self._arinyo_pars = np.array([q1, q2, kv, av, bv, kp]) + 1
        if not np.allclose(np.array([q1, q2, kv, av, bv, kp]), self._arinyo_pars):
            delta_squared = self.k_grid**3 * self._pk_fid / (2 * np.pi**2)
            growth = q1 * delta_squared + q2 * delta_squared**2
            pec_velocity = (self.k_grid / kv) ** av * np.abs(self.muk_grid) ** bv
            pressure = (self.k_grid / kp) * (self.k_grid / kp)
            dnl = np.exp(growth * (1 - pec_velocity) - pressure)

            if np.any(np.isnan(dnl)) or np.any(np.isinf(dnl)):
                raise utils.VegaArinyoError

            self._arinyo_pars = np.array([q1, q2, kv, av, bv, kp])
            if two_lya_flag:
                self._arinyo_dnl_cache = dnl
            elif one_lya_flag:
                self._arinyo_dnl_cache = np.sqrt(dnl)
            else:
                return np.ones(dnl.shape)

        return self._arinyo_dnl_cache

    def compute_dnl_forestflow(self, params):
        """Return an externally supplied ForestFlow nonlinear correction.

        Parameters
        ----------
        params : dict
            Must contain ``forestflow_dnl``. It can be one array on this
            component's ``(mu, k)`` grid, or a mapping keyed by data-set name.

        Returns
        -------
        numpy.ndarray
            The full Ly-alpha factor for Ly-alpha auto-correlations and its
            square root for a single Ly-alpha tracer. Metal-only correlations
            receive unity. If ``forestflow_metal_pressure`` is supplied (same
            grid, or a mapping by data-set name), each metal leg is further
            multiplied by it. If ``forestflow_metal_decorrelation`` (metal
            tracer name to line-of-sight sigma in Mpc/h) is supplied, Ly-alpha x
            metal correlations are multiplied by ``exp(-(k_par sigma)^2 / 2)``.
            SiII x SiII correlations (autos and crosses of SiII(1190), SiII(1193)
            and SiII(1260)) are multiplied by ``forestflow_metal_sii_factor``
            (``f > 0``) and by ``exp(-(k_par sigma)^2 / 2)`` with
            ``forestflow_metal_sii_decorrelation`` (``sigma >= 0`` in Mpc/h), the
            ``f_SiIIxSiII`` and ``sig_SiIIxSiII`` of the P1D Kaiser metal model;
            each is optional and leaves the model unchanged when absent.

        Notes
        -----
        The factor is deliberately supplied at the public ``chi2`` boundary.
        Vega keeps ownership of its tracer, Kaiser, metal and peak/smooth
        processing; it does not reimplement ForestFlow's kernel.
        """
        supplied = params.get("forestflow_dnl")
        if supplied is None:
            raise ValueError("small scale nl=forestflow requires forestflow_dnl")
        if isinstance(supplied, dict):
            if self._name not in supplied:
                raise ValueError(f"missing ForestFlow correction for {self._name!r}")
            supplied = supplied[self._name]
        dnl = np.asarray(supplied, dtype=float)
        expected = np.broadcast_shapes(self.muk_grid.shape, self.k_grid.shape)
        if dnl.shape != expected:
            raise ValueError(
                "ForestFlow nonlinear correction has shape "
                f"{dnl.shape}, expected Vega (mu, k) shape {expected}"
            )
        if not np.all(np.isfinite(dnl)):
            raise ValueError("ForestFlow nonlinear correction contains non-finite values")
        if np.any(dnl <= 0):
            raise ValueError(
                "ForestFlow nonlinear correction must be positive "
                f"(minimum={dnl.min():.3e}, maximum={dnl.max():.3e})"
            )
        two_lya = "LY" in self.tracer1_name and "LY" in self.tracer2_name
        one_lya = "LY" in self.tracer1_name or "LY" in self.tracer2_name
        if two_lya:
            factor = dnl
        elif one_lya:
            factor = np.sqrt(dnl)
        else:
            factor = np.ones_like(dnl)
        # Optional pressure smoothing of each metal leg (continuous tracers
        # other than Ly-alpha/beta), exp(-(k/k_p)^2 / 2) per leg, so that the
        # metal terms match the P1D Kaiser metal model.
        pressure = params.get("forestflow_metal_pressure")
        if pressure is not None:
            if isinstance(pressure, dict):
                if self._name not in pressure:
                    raise ValueError(f"missing ForestFlow metal pressure for {self._name!r}")
                pressure = pressure[self._name]
            pressure = np.asarray(pressure, dtype=float)
            if pressure.shape != expected or np.any(pressure <= 0) or np.any(pressure > 1):
                raise ValueError("ForestFlow metal pressure must be in (0, 1] on Vega's (mu, k) grid")
            metal_legs = sum(
                tracer_type == "continuous" and "LY" not in name
                for name, tracer_type in ((self.tracer1_name, self.tracer1_type),
                                          (self.tracer2_name, self.tracer2_type))
            )
            factor = factor * pressure**metal_legs
        # Optional line-of-sight decorrelation between Ly-alpha and a metal,
        # exp(-(k_par sigma)^2 / 2) with sigma in Mpc/h per metal tracer. It
        # acts only on Ly-alpha x metal; metal autos/crosses are unaffected.
        decorrelation = params.get("forestflow_metal_decorrelation")
        if decorrelation is not None and one_lya and not two_lya:
            other, other_type = (
                (self.tracer2_name, self.tracer2_type)
                if "LY" in self.tracer1_name
                else (self.tracer1_name, self.tracer1_type)
            )
            sigma = decorrelation.get(other) if other_type == "continuous" else None
            if sigma is not None:
                factor = factor * np.exp(-0.5 * (self.k_par_grid * float(sigma)) ** 2)
        # Optional amplitude factor and line-of-sight decorrelation of the SiII x SiII
        # correlations, decoupling them from the Ly-alpha x SiII biases as in the P1D Kaiser
        # metal model (f_SiIIxSiII, sig_SiIIxSiII): f exp(-(k_par sigma)^2 / 2), sigma in Mpc/h.
        if self._sii_x_sii:
            sii_factor, sii_sigma = sii_x_sii_parameters(params)
            if sii_factor is not None:
                factor = factor * sii_factor
            if sii_sigma is not None:
                factor = factor * np.exp(-0.5 * (self.k_par_grid * sii_sigma) ** 2)
        return factor

    def compute_Gk(self, params):
        """Model the effect of binning of the cf.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            G(k)
        """
        Gk = 1.0

        if self.rmu_binning:
            bin_size_r = params.get(f"r binsize {self._name}", self._bin_size_r)
            # bin_size_mu = params.get(f"mu binsize {self._name}", self._bin_size_mu)

            if bin_size_r != 0:
                Gk = Gk * utils.sinc(self.k_grid * bin_size_r / 2)
            # if bin_size_mu != 0:
            #     Gk = Gk * utils.sinc(self.muk_grid * bin_size_mu / 2)
        else:
            bin_size_rp = params.get("par binsize {}".format(self._name), self._bin_size_rp)
            bin_size_rt = params.get("per binsize {}".format(self._name), self._bin_size_rt)

            if bin_size_rp != 0:
                Gk = Gk * utils.sinc(self.k_par_grid * bin_size_rp / 2)
            if bin_size_rt != 0:
                Gk = Gk * utils.sinc(self.k_trans_grid * bin_size_rt / 2)

        return Gk

    def compute_fullshape_gauss_smoothing(self, params):
        """Compute a Gaussian smoothing for the full correlation function.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        check_tracer1 = self.tracer1_name in ["LYA", "QSO"]
        check_tracer2 = self.tracer2_name in ["LYA", "QSO"]

        if ("par_sigma_smooth" in params) or ("per_sigma_smooth" in params):
            sigma_par = params.get("par_sigma_smooth", None)
            sigma_trans = params.get("per_sigma_smooth", None)

            if sigma_par is None and sigma_trans is None:
                raise ValueError(
                    "Asked for fullshape gaussian smoothing without setting the"
                    " smoothing parameters (par_sigma_smooth and/or per_sigma_smooth)."
                )
            elif sigma_par is None:
                sigma_par = sigma_trans
            elif sigma_trans is None:
                sigma_trans = sigma_par

            return (
                utils.compute_gauss_smoothing(
                    sigma_par, sigma_trans, self.k_par_grid, self.k_trans_grid
                )
                ** 2
            )

        elif (
            ("par_sigma_smooth_metals" in params)
            and ("per_sigma_smooth_metals" in params)
            and not (check_tracer1 and check_tracer2)
        ):
            return (
                utils.compute_gauss_smoothing(
                    params["par_sigma_smooth_metals"],
                    params["per_sigma_smooth_metals"],
                    self.k_par_grid,
                    self.k_trans_grid,
                )
                ** 2
            )

        else:
            return utils.compute_gauss_smoothing(
                params[f"par_sigma_smooth_{self.tracer1_name}"],
                params[f"per_sigma_smooth_{self.tracer1_name}"],
                self.k_par_grid,
                self.k_trans_grid,
            ) * utils.compute_gauss_smoothing(
                params[f"par_sigma_smooth_{self.tracer2_name}"],
                params[f"per_sigma_smooth_{self.tracer2_name}"],
                self.k_par_grid,
                self.k_trans_grid,
            )
        # else:
        #     raise ValueError(
        #         'Asked for fullshape gaussian smoothing without correctly setting the'
        #         ' smoothing parameters (par_sigma_smooth and/or per_sigma_smooth).'
        #     )

    def compute_fullshape_exp_smoothing(self, params):
        """Compute a Gaussian and exp smoothing for the full
        correlation function (usefull for london_mocks_v6.0).

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        # Get the parameters
        sigma_par_sq = params["par_sigma_smooth"] ** 2
        sigma_trans_sq = params["per_sigma_smooth"] ** 2
        exp_par_sq = params["par_exp_smooth"] ** 2
        exp_trans_sq = params["per_exp_smooth"] ** 2

        # Compute the smoothing components
        gauss_smoothing = self.k_par_grid**2 * sigma_par_sq
        gauss_smoothing += self.k_trans_grid**2 * sigma_trans_sq
        exp_smoothing = np.abs(self.k_par_grid) * exp_par_sq
        exp_smoothing += np.abs(self.k_trans_grid) * exp_trans_sq

        return np.exp(-gauss_smoothing / 2) * np.exp(-exp_smoothing)

    def compute_velocity_dispersion_gauss(self, params):
        """Compute a gaussian smoothing factor to model velocity dispersion.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        assert "discrete" in [self.tracer1_type, self.tracer2_type]

        smoothing = np.ones(self.k_par_grid.shape)
        if self.tracer1_type == "discrete":
            sigma = params["sigma_velo_disp_gauss_" + self.tracer1_name]
            smoothing *= np.exp(-0.25 * (self.k_par_grid * sigma) ** 2)
        if self.tracer2_type == "discrete":
            sigma = params["sigma_velo_disp_gauss_" + self.tracer2_name]
            smoothing *= np.exp(-0.25 * (self.k_par_grid * sigma) ** 2)

        return smoothing

    def compute_velocity_dispersion_lorentz(self, params):
        """Compute a lorentzian smoothing factor to model velocity dispersion.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        assert "discrete" in [self.tracer1_type, self.tracer2_type]

        smoothing = np.ones(self.k_par_grid.shape)
        if self.tracer1_type == "discrete":
            sigma = params["sigma_velo_disp_lorentz_" + self.tracer1_name]
            smoothing *= 1.0 / np.sqrt(1 + (self.k_par_grid * sigma) ** 2)
        if self.tracer2_type == "discrete":
            sigma = params["sigma_velo_disp_lorentz_" + self.tracer2_name]
            smoothing *= 1.0 / np.sqrt(1 + (self.k_par_grid * sigma) ** 2)

        return smoothing
