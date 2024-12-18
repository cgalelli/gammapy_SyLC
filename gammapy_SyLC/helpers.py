import inspect
import numpy as np
from multiprocessing import Pool
from scipy.signal import periodogram
from scipy.interpolate import interp1d

from .simulators import TimmerKonig_lightcurve_simulator, Emmanoulopoulos_lightcurve_simulator


def _generate_periodogram(args):
    """Helper function to generate a periodogram for a single realization for multiprocessing."""
    (
        simulator,
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params,
        psd_params,
        mean,
        std,
        noise,
        noise_type,
    ) = args
    if simulator == "TK":
        tseries, _ = TimmerKonig_lightcurve_simulator(
            psd,
            npoints,
            spacing,
            power_spectrum_params=psd_params,
            mean=mean,
            std=std,
            noise=noise,
            noise_type=noise_type,
        )
    elif simulator == "EMM":
        tseries, _ = Emmanoulopoulos_lightcurve_simulator(
            pdf,
            psd,
            npoints,
            spacing,
            pdf_params=pdf_params,
            psd_params=psd_params,
            mean=mean,
            std=std,
            noise=noise,
            noise_type=noise_type,
        )
    else:
        raise ValueError("Invalid simulator. Use 'TK' or 'EMM'.")

    freqs, pg = periodogram(tseries, 1 / spacing.value)
    return pg


def _wrap_emm(args):
    """Helper wrapper of the Emmanoulopoulos algorithm for multiprocessing."""
    (
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params,
        psd_params,
        mean,
        std,
        noise,
        noise_type,
    ) = args
    tseries, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
        noise=noise,
        noise_type=noise_type,
    )
    return tseries


def lightcurve_psd_envelope(
        psd,
        npoints,
        spacing,
        pdf=None,
        nsims=10000,
        pdf_params=None,
        psd_params=None,
        simulator="TK",
        mean=0.0,
        std=1.0,
        oversample=10,
        noise=None,
        noise_type="gauss",
):
    """
    Generate PSD envelopes for light curves simulated using Timmer & Koenig (TK)
    or Emmanoulopoulos (EMM) algorithms.

    Parameters:
    -----------
    psd : callable
        Target power spectral density function.
    npoints : int
        Number of points in the simulated light curve.
    spacing : astropy.units.Quantity
        Time spacing between successive points.
    pdf : callable or None, optional
        Probability density function for flux amplitudes. Required if `simulator="EMM"`.
    nsims : int, optional
        Number of simulations to generate for the envelope. Default is 10000.
    pdf_params : dict, optional
        Parameters for the PDF function. Default is None.
    psd_params : dict, optional
        Parameters for the PSD function. Default is None.
    simulator : {'TK', 'EMM'}, optional
        Simulator to use ('TK' for Timmer & Koenig or 'EMM' for Emmanoulopoulos). Default is 'TK'.
    mean : float, optional
        Desired mean of the light curve. Default is 0.0.
    std : float, optional
        Desired standard deviation of the light curve. Default is 1.0.
    oversample : int, optional
        Oversampling factor for the light curves. Default is 10.
    noise : float or None, optional
        Noise amplitude to add to the light curve. Default is None.
    noise_type : {'gauss', 'counts'}, optional
        Type of noise to add. Default is 'gauss'.

    Returns:
    --------
    envelopes_psd : ndarray
        Array of PSD values from the simulated light curves.
    freqs : ndarray
        Frequencies corresponding to the PSD values.
    """
    npoints_ext = npoints * oversample
    spacing_ext = spacing / oversample

    args = [
        (
            simulator,
            pdf,
            psd,
            npoints_ext,
            spacing_ext,
            pdf_params,
            psd_params,
            mean,
            std,
            noise,
            noise_type,
        )
        for _ in range(nsims)
    ]

    with Pool() as pool:
        results = pool.map(_generate_periodogram, args)

    envelopes_psd = np.array(results)[..., 1: npoints // 2 + 1]

    freqs = np.fft.fftfreq(npoints_ext, spacing_ext.value)[1: npoints // 2 + 1]

    return envelopes_psd, freqs


def _psd_fit_helper(
        psd_params_list,
        pgram,
        npoints,
        spacing,
        psd,
        pdf=None,
        pdf_params=None,
        simulator="TK",
        nsims=10000,
        mean=None,
        std=None,
        noise=None,
        noise_type="gauss",
):
    psd_params_keys = list(inspect.signature(psd).parameters.keys())

    if len(psd_params_keys[1:]) != len(psd_params_list):
        raise ValueError(
            "parameter values do not correspond to the request from the psd function"
        )

    psd_params = dict(zip(psd_params_keys[1:], psd_params_list))

    envelopes, freqs = lightcurve_psd_envelope(
        psd,
        npoints,
        spacing,
        pdf=pdf,
        pdf_params=pdf_params,
        psd_params=psd_params,
        simulator=simulator,
        nsims=nsims,
        mean=mean,
        std=std,
        noise=noise,
        noise_type=noise_type,
    )

    if len(envelopes[0]) != len(pgram):
        raise ValueError("required length is different than data length!")

    obs = (pgram - np.nanmean(envelopes, axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sim = (envelopes - np.nanmean(envelopes, axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sumobs = np.sum(obs)
    sumsim = np.sum(sim, axis=-1)
    sign = len(np.where(sumobs >= sumsim)[0]) / nsims
    return sumobs * sign / len(obs)


def _pdf_fit_helper(
        pdf_params_list,
        flux,
        npoints,
        spacing,
        psd,
        psd_params,
        pdf,
        nsims=500,
        mean=None,
        std=None,
        noise=None,
        noise_type="gauss",
):
    pdf_params_keys = list(inspect.signature(pdf).parameters.keys())

    if len(pdf_params_keys[1:]) != len(pdf_params_list):
        raise ValueError(
            "parameter values do not correspond to the request from the pdf function"
        )

    pdf_params = dict(zip(pdf_params_keys[1:], pdf_params_list))

    args = [
        (
            pdf,
            psd,
            npoints,
            spacing,
            pdf_params,
            psd_params,
            mean,
            std,
            noise,
            noise_type,
        )
        for _ in range(nsims)
    ]

    with Pool() as pool:
        results = pool.map(_wrap_emm, args)

    hist, bin_edges = np.histogram(np.array(results).flatten(), bins=int(npoints * nsims / 10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    pdf_interpolated = interp1d(bin_centers, hist, kind="cubic", fill_value="extrapolate")

    likelihoods = np.maximum(pdf_interpolated(flux), 1e-10)
    nll = -np.sum(np.log(likelihoods))

    return nll
