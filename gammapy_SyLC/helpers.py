import inspect
import numpy as np
from astropy.timeseries import LombScargle, LombScargleMultiband
from scipy.interpolate import PchipInterpolator
from .simulators import TimmerKonig_lightcurve_simulator, ModifiedTimmerKonig_lightcurve_simulator, Emmanoulopoulos_lightcurve_simulator

def lightcurve_psd_envelope(
        psd,
        obs_times,
        frequencies=None,
        pdf=None,
        nsims=10000,
        pdf_params=None,
        psd_params=None,
        simulator="MTK",
        mean=0.0,
        std=1.0,
        known_times=None,
        known_fluxes=None,
        bands=None,
        flux_error=None,
):
    """
    Generate PSD envelopes for light curves simulated using Timmer & Koenig (TK)
    or Emmanoulopoulos (EMM) algorithms.

    Parameters:
    -----------
    psd : callable
        Target power spectral density function.
    obs_times : astropy.units.Quantity
        Observation times. Needs to be evenly spaced if simulator `simulator="EMM"` or '"TK"'.
    frequencies : ndarray or None, optional
        Frequencies at which to compute the periodogram. If None, frequencies are
        automatically determined. Default is None.
    pdf : callable or None, optional
        Probability density function for flux amplitudes. Required if `simulator="EMM"`.
    nsims : int, optional
        Number of simulations to generate for the envelope. Default is 10000.
    pdf_params : dict, optional
        Parameters for the PDF function. Default is None.
    psd_params : dict, optional
        Parameters for the PSD function. Default is None.
    simulator : {'TK', 'MTK', 'EMM'}, optional
        Simulator to use ('TK' for Timmer & Koenig or 'MTK' for the modified Timmer & Koenig or 'EMM' for Emmanoulopoulos). Default is 'TK'.
    mean : float, optional
        Desired mean of the light curve. Default is 0.0.
    std : float, optional
        Desired standard deviation of the light curve. Default is 1.0.
    known_times : ndarray or None, optional
        Times of known flux measurements to include in the periodogram. Default is None.
    known_fluxes : ndarray or None, optional
        Known flux measurements corresponding to `known_times`. Default is None.
    bands : ndarray or None, optional
        Band identifiers for multiband Lomb-Scargle periodogram. Length must match `
    oversample : int, optional
        Oversampling factor for the light curves. Default is 10.
    noise : float or None, optional
        Noise (relative) amplitude to add to the light curve. Default is None.

    Returns:
    --------
    envelopes_psd : ndarray
        Array of PSD values from the simulated light curves.
    freqs : ndarray
        Frequencies corresponding to the PSD values.
    """
    if not np.allclose(np.diff(obs_times), np.diff(obs_times)[0], rtol=1e-5) and simulator != "MTK":
            raise ValueError("Using an unevenly sampled 'obs_times' with a simulator that does not support it. Use simulator='MTK' for uneven observation times.")

    if simulator == "TK":
        tseries_matrix, taxis = TimmerKonig_lightcurve_simulator(
            psd, obs_times, psd_params=psd_params, mean=mean, std=std, nsims=nsims
        )
    elif simulator == "MTK":
        tseries_matrix, taxis = ModifiedTimmerKonig_lightcurve_simulator(
            psd, obs_times, psd_params=psd_params, mean=mean, std=std, nsims=nsims
        )
    elif simulator == "EMM":
        tseries_matrix, taxis = Emmanoulopoulos_lightcurve_simulator(
            pdf, psd, obs_times, pdf_params=pdf_params, psd_params=psd_params, mean=mean, std=std, nsims=nsims
        )
    else:
        raise ValueError("Invalid simulator. Use 'TK', 'MTK' or 'EMM'.")

    all_pgs = []
    

    if known_times is None and known_fluxes is None and bands is None: 
        for i in range(nsims):
            ls = LombScargle(taxis, tseries_matrix[i], flux_error)
            if frequencies is not None:
                power = ls.power(frequencies, normalization="psd")
            else:
                frequencies, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")
            all_pgs.append(power)
    elif known_times is not None and known_fluxes is not None and bands is not None:
        full_times = np.concatenate((known_times, taxis))
        full_fluxes = np.concatenate((known_fluxes, tseries_matrix), axis=0)
        full_bands = np.concatenate((bands, np.zeros_like(taxis, dtype=int)))

        for i in range(nsims):
            ls = LombScargleMultiband(full_times, full_fluxes[i], full_bands)
            if frequencies is not None:
                power = ls.power(frequencies, normalization="psd")
            else:
                frequencies, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")
            all_pgs.append(power)

    else:
        raise ValueError("If known_times, known_fluxes and bands are not None, all of them must be provided.")

    envelopes_psd = np.array(all_pgs)
    return envelopes_psd, frequencies

def interp_pdf(
        psd,
        pdf,
        psd_params,
        pdf_params,
        obs_times,
        nsims=1000,
        mean=0.0,
        std=1.0,
):
    """
    Generate an interpolated probability density function (PDF) for flux amplitudes
    based on Monte Carlo simulations using the Emmanoulopoulos algorithm.

    Parameters:
    -----------
    psd : callable
        Target power spectral density (PSD) function to simulate the light curves.
    pdf : callable
        Target probability density function (PDF) for flux amplitudes.
    psd_params : dict
        Parameters for the PSD function.
    pdf_params : dict
        Parameters for the PDF function.
    obs_times : astropy.units.Quantity
        Observation times. Needs to be evenly spaced.
    nsims : int, optional
        Number of Monte Carlo simulations to generate. Default is 10000.
    mean : float, optional
        Desired mean of the simulated light curves. Default is 0.0.
    std : float, optional
        Desired standard deviation of the simulated light curves. Default is 1.0.

    Returns:
    --------
    pdf_interpolated : PchipInterpolator
        A piecewise cubic Hermite interpolating polynomial (PCHIP) object representing
        the observed PDF based on the simulated flux amplitudes.
    """
    tseries_matrix, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf, psd, obs_times, pdf_params=pdf_params, psd_params=psd_params, mean=mean, std=std, nsims=nsims
    )

    hist, bin_edges = np.histogram(tseries_matrix.flatten(), bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    pdf_interpolated = PchipInterpolator(bin_centers, hist)
    return pdf_interpolated


def _psd_fit_helper(
        psd_params_list,
        frequencies,
        power,
        obs_times,
        psd,
        pdf=None,
        pdf_params=None,
        simulator="TK",
        nsims=10000,
        mean=0.,
        std=1.,
        known_times=None,
        known_fluxes=None,
        bands=None,
        flux_error=None,
):
    psd_params_keys = list(inspect.signature(psd).parameters.keys())

    if len(psd_params_keys[1:]) != len(psd_params_list):
        raise ValueError(
            "parameter values do not correspond to the request from the psd function"
        )

    psd_params = dict(zip(psd_params_keys[1:], psd_params_list))

    envelopes, freqs = lightcurve_psd_envelope(
        psd,
        obs_times,
        frequencies=frequencies, 
        pdf=pdf,
        pdf_params=pdf_params,
        psd_params=psd_params,
        simulator=simulator,
        nsims=nsims,
        mean=mean,
        std=std,
        known_times=known_times,
        known_fluxes=known_fluxes,
        bands=bands,
        flux_error=flux_error,
    )

    if len(envelopes[0]) != len(power):
        raise ValueError("required length is different than data length!")

    obs = (power - np.nanmean(envelopes, axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sim = (envelopes - np.nanmean(envelopes, axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sumobs = np.sum(obs)
    sumsim = np.sum(sim, axis=-1)
    sign = len(np.where(sumobs >= sumsim)[0]) / nsims
    return sumobs * sign / len(obs)


def _pdf_fit_helper(
        pdf_params_list,
        flux,
        obs_times,
        psd,
        psd_params,
        pdf,
        nsims=500,
        flux_error=None,
):
    mean = np.mean(flux)
    std = np.std(flux)

    param_names = list(inspect.signature(pdf).parameters.keys())[1:] 
    fit_param_names = [p for p in param_names if p not in ['mean', 'std']]
    pdf_params = dict(zip(fit_param_names, pdf_params_list))

    pdf_interpolated = interp_pdf(
        psd,
        pdf,
        psd_params,
        pdf_params,
        obs_times,
        nsims=nsims,
        mean=mean,
        std=std,
    )

    if flux_error is not None:
        likelihoods = np.prod(np.maximum(pdf_interpolated(np.random.normal(flux[:, None], flux_error[:, None], (len(flux), 10))), 1), axis=-1) ** (1 / 10)
    else:
        likelihoods = np.maximum(pdf_interpolated(flux), 1)

    nll = -np.sum(np.log(likelihoods))

    return nll
