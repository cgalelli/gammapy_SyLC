import inspect
import numpy as np
from scipy.optimize import minimize

from .helpers import _pdf_fit_helper, _psd_fit_helper
from .simulators import Emmanoulopoulos_lightcurve_simulator


def psd_fit(
        pgram,
        psd,
        psd_initial,
        spacing,
        pdf=None,
        pdf_params=None,
        simulator="TK",
        nsims=10000,
        mean=None,
        std=None,
        noise=None,
        noise_type="gauss",
        nexp=50,
        full_output=False,
        **kwargs,
):
    """
    Fit a power spectral density (PSD) model to an observed periodogram using
    simulated light curves and Monte Carlo optimization.

    Parameters:
    -----------
    pgram : ndarray
        Observed periodogram values representing the data.
    psd : callable
        Target power spectral density (PSD) model function.
    psd_initial : dict
        Initial guesses for the PSD model parameters.
    spacing : astropy.units.Quantity
        Time spacing for the light curve.
    pdf : callable or None, optional
        Target probability density function (PDF) for flux amplitudes, required for
        Emmanoulopoulos (EMM) simulations. Default is None.
    pdf_params : dict or None, optional
        Parameters for the PDF function. Default is None.
    simulator : {'TK', 'EMM'}, optional
        Simulator to use ('TK' for Timmer & Koenig or 'EMM' for Emmanoulopoulos). Default is 'TK'.
    nsims : int, optional
        Number of simulations for envelope generation. Default is 10000.
    mean : float or None, optional
        Desired mean for the simulated light curves. Default is None.
    std : float or None, optional
        Desired standard deviation for the simulated light curves. Default is None.
    noise : float or None, optional
        Noise amplitude to add to the simulated light curves. Default is None.
    noise_type : {'gauss', 'counts'}, optional
        Type of noise to add to the simulated light curves. Default is 'gauss'.
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
            pgram[1:],
            len(pgram[1:]) * 2,
            spacing,
            psd,
            pdf,
            pdf_params,
            simulator,
            nsims,
            mean,
            std,
            noise,
            noise_type,
        ),
        **kwargs,
    )
    psd_params_keys = list(inspect.signature(psd).parameters.keys())
    psd_params = dict(zip(psd_params_keys[1:], results.x))

    if nexp > 0:
        results_list = np.empty((nexp,) + results.x.shape)
        frequencies = np.fft.fftfreq(len(pgram), spacing.value)
        real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))
        test_pgram = psd(real_frequencies, **psd_params)

        for _ in range(nexp):
            results_err = psd_fit(
                test_pgram,
                psd,
                psd_params,
                spacing,
                pdf=pdf,
                pdf_params=pdf_params,
                simulator=simulator,
                nsims=100,
                mean=mean,
                std=std,
                nexp=-1,
                **kwargs,
            )
            results_list[_] = results_err
        error = np.std(results_list, axis=0)

        if full_output:
            return results, error
        else:
            return results.x, error

    else:
        if full_output:
            return results
        else:
            return results.x


def pdf_fit(
        flux,
        psd,
        psd_params,
        pdf,
        pdf_initial,
        spacing,
        nsims=10000,
        mean=None,
        std=None,
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
    psd : callable
        Target power spectral density (PSD) function.
    psd_params : dict
        Parameters for the PSD function.
    pdf : callable
        Target probability density function (PDF) for flux amplitudes.
    pdf_initial : dict
        Initial guesses for the PDF parameters.
    spacing : astropy.units.Quantity
        Time spacing for the light curve.
    nsims : int, optional
        Number of Monte Carlo simulations for the fitting procedure. Default is 10000.
    mean : float or None, optional
        Desired mean for the simulated light curves. Default is None.
    std : float or None, optional
        Desired standard deviation for the simulated light curves. Default is None.
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
    kwargs.setdefault("method", "Powell")
    results = minimize(
        _pdf_fit_helper,
        list(pdf_initial.values()),
        args=(
            flux,
            len(flux),
            spacing,
            psd,
            psd_params,
            pdf,
            nsims,
            mean,
            std,
            flux_error,
        ),
        **kwargs,
    )

    if output_type == "parameters":
        return results.x
    elif output_type == "value":
        return results.fun
    elif output_type == "full":
        return results
    else:
        raise ValueError(
            f"Accepted values for {output_type} are 'parameters', 'value' or 'full'"
        )


def compare_pdf_models(
        flux,
        pdf_test,
        pdf_initial,
        psd,
        psd_params,
        pdf,
        pdf_params,
        spacing,
        nsims=1000,
        ntests=100,
        mean=None,
        std=None,
        flux_error=None,
        **kwargs,
):

    fit_stats = pdf_fit(
        flux,
        psd,
        psd_params,
        pdf_test,
        pdf_initial,
        spacing,
        nsims=nsims,
        mean=mean,
        std=std,
        flux_error=flux_error,
        output_type="value",
        **kwargs,)

    num = []
    for j in range(ntests):
        tseries, _ = Emmanoulopoulos_lightcurve_simulator(
            pdf,
            psd,
            len(flux),
            spacing,
            pdf_params=pdf_params,
            psd_params=psd_params,
            mean=mean,
            std=std,
        )
        fit_test = pdf_fit(
            tseries,
            psd,
            psd_params,
            pdf_test,
            pdf_initial,
            spacing,
            nsims=nsims,
            mean=mean,
            std=std,
            flux_error=flux_error,
            output_type="value",
            **kwargs,)
        print(fit_test, j)
        num.append(fit_test - fit_stats)

    return num