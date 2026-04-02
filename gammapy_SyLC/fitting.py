import inspect
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool

from .helpers import _pdf_fit_helper, _psd_fit_helper
from .simulators import Emmanoulopoulos_lightcurve_simulator, TimmerKonig_lightcurve_simulator
from .distributions import get_physical_bounds

def _psd_fit_error_worker(args):
    """Worker to run a single PSD fit iteration for error estimation."""
    (
        frequencies, test_pgram, obs_times, psd, psd_params, pdf, pdf_params,
        simulator, mean, std, known_times, known_fluxes, bands, flux_error, kwargs
    ) = args
    
    return psd_fit(
        frequencies=frequencies,
        power=test_pgram,
        obs_times=obs_times,
        psd=psd,
        psd_initial=psd_params,
        pdf=pdf,
        pdf_params=pdf_params,
        simulator=simulator,
        nsims=100,
        mean=mean,
        std=std,
        known_times=known_times,
        known_fluxes=known_fluxes,
        bands=bands,
        flux_error=flux_error,
        nexp=-1,
        **kwargs,
    )

def _compare_normal_worker(args):
    """Worker to simulate a normal light curve and fit it."""
    (
        obs_times, pdf_test, pdf_initial, psd, psd_params, mean, std, nsims, flux_error, kwargs
    ) = args
    
    tseries, _ = TimmerKonig_lightcurve_simulator(
        power_spectrum=psd,
        obs_times=obs_times,
        psd_params=psd_params,
        mean=mean,
        std=std,
        nsims=1,
    )
    
    fit_val = pdf_fit(
        flux=tseries,
        obs_times=obs_times,
        psd=psd,
        psd_params=psd_params,
        pdf=pdf_test,
        pdf_initial=pdf_initial,
        nsims=nsims,
        flux_error=flux_error,
        output_type="value",
        **kwargs,
    )
    return fit_val


def _compare_models_worker(args):
    """Worker to simulate an Emmanoulopoulos light curve and fit it."""
    (
        obs_times, pdf_test, pdf_initial, psd, psd_params, pdf, pdf_params, 
        mean, std, nsims, flux_error, kwargs
    ) = args
    
    # 1. Simulate using Emmanoulopoulos (Underlying test distribution)
    tseries, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf=pdf,
        psd=psd,
        obs_times=obs_times,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
        nsims=1,
    )
    
    fit_val = pdf_fit(
        flux=tseries,
        psd=psd,
        psd_params=psd_params,
        pdf=pdf_test,
        pdf_initial=pdf_initial,
        obs_times=obs_times,
        nsims=nsims,
        flux_error=flux_error,
        output_type="value",
        **kwargs,
    )
    return fit_val


def psd_fit(
        frequencies,
        power,
        obs_times,
        psd,
        psd_initial,
        pdf=None,
        pdf_params=None,
        simulator="MTK",
        nsims=10000,
        mean=None,
        std=None,
        known_times=None,
        known_fluxes=None,
        bands=None,
        flux_error=None,
        nexp=50,
        full_output=False,
        **kwargs,
):
    """
    Fit a power spectral density (PSD) model to an observed periodogram using
    simulated light curves and Monte Carlo optimization.

    Parameters:
    -----------
    frequencies : ndarray
        Frequencies of sampling of the periodogram.
    power : ndarray
        Observed periodogram values representing the data.
    obs_times : astropy.units.Quantity
        Observation times for the light curve.
    psd : callable
        Target power spectral density (PSD) model function.
    psd_initial : dict
        Initial guesses for the PSD model parameters.
    pdf : callable or None, optional
        Target probability density function (PDF) for flux amplitudes, required for
        Emmanoulopoulos (EMM) simulations. Default is None.
    pdf_params : dict or None, optional
        Parameters for the PDF function. Default is None.
    simulator : {'TK', 'MTK', 'EMM'}, optional
        Simulator to use ('TK' for Timmer & Koenig or 'MTK' for the modified Timmer & Koenig or 'EMM' for Emmanoulopoulos). Default is 'TK'.
    nsims : int, optional
        Number of simulations for envelope generation. Default is 10000.
    mean : float or None, optional
        Desired mean for the simulated light curves. Default is None.
    std : float or None, optional
        Desired standard deviation for the simulated light curves. Default is None.
    known_times : ndarray or None, optional
        Times of known flux measurements to include in the periodogram. Default is None.
    known_fluxes : ndarray or None, optional
        Known flux measurements corresponding to `known_times`. Default is None.
    bands : ndarray or None, optional
        Band identifiers for multiband Lomb-Scargle periodogram. Length must match `len(known_times) + len(obs_times)`. Default is None.
    flux_error : ndarray or None, optional
        Observed flux uncertainties to be used to account for measurement errors in the
    nexp : int, optional
        Number of Monte Carlo simulations for uncertainty estimation. Default is 50.
    full_output : bool, optional
        If True, returns the full optimization result. Default is False.
    **kwargs : dict
        Additional keyword arguments for the optimizer.

    Returns:
    --------
    results : OptimizeResult or ndarray
        The optimization result if `full_output` is True, otherwise the best-fit parameters.
    error : ndarray, optional
        Uncertainties in the estimated parameters (if `nexp > 0`).
    """
    if not isinstance(nexp, int):
        raise TypeError(
            "The number of MC simulations for the error evaluation nexp must be an integer!"
        )
    kwargs.setdefault("method", "Powell")

    results = minimize(
        _psd_fit_helper,
        list(psd_initial.values()),
        args=(
            frequencies, power, obs_times, psd, pdf, pdf_params,
            simulator, nsims, mean, std, known_times, known_fluxes, bands, flux_error,
        ),
        **kwargs,
    )
    
    psd_params_keys = list(inspect.signature(psd).parameters.keys())
    psd_params = dict(zip(psd_params_keys[1:], results.x))

    if nexp > 0:
        test_pgram = psd(frequencies, **psd_params).value * power.unit
        
        worker_args = [
            (
                frequencies, test_pgram, obs_times, psd, psd_params, pdf, pdf_params,
                simulator, mean, std, known_times, known_fluxes, bands, flux_error, kwargs
            )
            for _ in range(nexp)
        ]
        
        with Pool() as pool:
            results_list = pool.map(_psd_fit_error_worker, worker_args)
            
        results_array = np.array(results_list)
        error = np.std(results_array, axis=0)

        return (results, error) if full_output else (results.x, error)

    return results if full_output else results.x


def pdf_fit(
        flux,
        obs_times,
        psd,
        psd_params,
        pdf,
        pdf_initial,
        nsims=10000,
        flux_error=None,
        output_type="value",
        **kwargs,
):
    """
    Fit a probability density function (PDF) to unbinned data
    using Monte Carlo simulations and a maximum likelihood approach.

    Parameters:
    -----------
    flux : ndarray
        Observed flux values representing the data.
    obs_times : astropy.units.Quantity
        Observation times for the light curve.
    psd : callable
        Target power spectral density (PSD) function.
    psd_params : dict
        Parameters for the PSD function.
    pdf : callable
        Target probability density function (PDF) for flux amplitudes.
    pdf_initial : dict
        Initial guesses for the PDF parameters.
    nsims : int, optional
        Number of Monte Carlo simulations for the fitting procedure. Default is 10000.
    flux_error : ndarray
        Observed flux uncertainties to be used to account for measurement errors in the fit. Default is None.
    output_type : string, optional
        Change output type. Can be "full", "parameters", "value". Default is "value".
    **kwargs : dict
        Additional keyword arguments for the optimizer.

    Returns:
    --------
    results : OptimizeResult or ndarray
        The optimization result if output_type="full", otherwise the best-fit parameters or function value.
    error : ndarray, optional
        Uncertainties in the estimated parameters (if `nexp > 0`).
    """
    mean = np.mean(flux)
    std = np.std(flux)

    if "bounds" not in kwargs:
        if pdf.__name__ == "lognormal":
            s_min = get_physical_bounds(mean, std, "lognormal")
            kwargs["bounds"] = [(s_min + 1e-4, None)]
        elif pdf.__name__ == "gammaf":
            a_max = get_physical_bounds(mean, std, "gamma")
            kwargs["bounds"] = [(1e-4, a_max - 1e-4)]

    kwargs.setdefault("method", "Powell")

    results = minimize(
        _pdf_fit_helper,
        list(pdf_initial.values()),
        args=(flux, obs_times, psd, psd_params, pdf, nsims, flux_error),
        **kwargs,
    )

    if output_type == "parameters":
        return results.x
    elif output_type == "value":
        return results.fun
    return results


def compare_normal( # noqa
        flux,
        obs_times,
        pdf_test,
        pdf_initial,
        psd,
        psd_params,
        nsims=100,
        ntests=100,
        flux_error=None,
        verbose=False,
        **kwargs,
):
    """
    Evaluates the significance of a fitted probability density function (PDF)
    by comparing it against a set of normally distributed synthetic light curves
    generated using the Timmer & König algorithm.

    Parameters
    ----------
    flux : array-like
        The observed light curve flux values.
    obs_times : astropy.units.Quantity
        Observation times for the light curve.
    pdf_test : callable
        The PDF model to be tested.
    pdf_initial : dict
        Initial parameter values for the PDF model.
    psd : callable
        The power spectral density (PSD) model used for simulation.
    psd_params : dict
        Parameters for the PSD model.
    nsims : int, optional (default=100)
        Number of simulated light curves to generate for the fit.
    ntests : int, optional (default=100)
        Number of Monte Carlo trials to assess statistical significance.
    flux_error : array-like, optional
        Measurement errors associated with the flux values.
    verbose : bool
        Flag to print partial results if True. Default is False.
    **kwargs : dict
        Additional arguments passed to `pdf_fit`.

    Returns
    -------
    num : list of float
        List of differences between the test statistic of the real data and
        those obtained from synthetic light curves.
    """
    mean = flux.mean()
    std = flux.std()

    fit_stats = pdf_fit(
        flux=flux,
        obs_times=obs_times,
        psd=psd,
        psd_params=psd_params,
        pdf=pdf_test,
        pdf_initial=pdf_initial,
        nsims=nsims*5,
        flux_error=flux_error,
        output_type="full",
        **kwargs, 
    )

    if verbose: 
        print(f"Base fit status: {fit_stats.message}")

    worker_args = [
        (obs_times, pdf_test, pdf_initial, psd, psd_params, mean, std, nsims, flux_error, kwargs)
        for _ in range(ntests)
    ]

    with Pool() as pool:
        test_funs = pool.map(_compare_normal_worker, worker_args)

    test_funs = np.array(test_funs)
    num = test_funs - fit_stats.fun

    p_value = len(num[num < 0]) / ntests
    return fit_stats, p_value


def compare_models( # noqa
        flux,
        obs_times,
        pdf_test,
        pdf_initial,
        psd,
        psd_params,
        pdf,
        pdf_params,
        nsims=100,
        ntests=100,
        flux_error=None,
        verbose=False,
        **kwargs,
):
    """
    Compares the significance of a fitted probability density function (PDF)
    using light curves generated with the Emmanoulopoulos algorithm with a
    different underlying distribution

    Parameters
    ----------
    flux : array-like
        The observed light curve flux values.
    obs_times : astropy.units.Quantity
        Observation times for the light curve.
    pdf_test : callable
        The PDF model to be tested.
    pdf_initial : dict
        Initial parameter values for the PDF model.
    psd : callable
        The power spectral density (PSD) model used for simulation.
    psd_params : dict
        Parameters for the PSD model.
    pdf : callable
        The PDF model used for generating synthetic light curves.
    pdf_params : dict
        Parameters for the PDF model used in simulations.
    nsims : int, optional (default=100)
        Number of simulated light curves to generate for the fit.
    ntests : int, optional (default=100)
        Number of Monte Carlo trials to assess statistical significance.
    flux_error : array-like, optional
        Measurement errors associated with the flux values.
    verbose : bool
        Flag to print partial results if True. Default is False.
    **kwargs : dict
        Additional arguments passed to `pdf_fit`.

    Returns
    -------
    num : list of float
        List of differences between the test statistic of the real data and
        those obtained from synthetic light curves.
    """
    mean = flux.mean()
    std = flux.std()

    fit_stats = pdf_fit(
        flux=flux,
        psd=psd,
        psd_params=psd_params,
        pdf=pdf_test,
        pdf_initial=pdf_initial,
        obs_times=obs_times,
        nsims=nsims*5,
        flux_error=flux_error,
        output_type="full",
        **kwargs, 
    )
    
    if verbose: 
        print(f"Base fit status: {fit_stats.message}")

    worker_args = [
        (obs_times, pdf_test, pdf_initial, psd, psd_params, pdf, pdf_params, mean, std, nsims, flux_error, kwargs)
        for _ in range(ntests)
    ]

    with Pool() as pool:
        test_funs = pool.map(_compare_models_worker, worker_args)

    test_funs = np.array(test_funs)
    num = test_funs - fit_stats.fun

    p_value = len(num[num < 0]) / ntests
    return fit_stats, p_value
