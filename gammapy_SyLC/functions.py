import numpy as np
import inspect
from gammapy.utils.random import get_random_state
from scipy.stats import gamma, lognorm
from scipy.signal import periodogram
from scipy.optimize import minimize
from multiprocessing import Pool


def lognormal(x, s):
    return lognorm.pdf(x, s, loc=0, scale=1)


def gammaf(x, a):
    return gamma.pdf(x, a, loc=0, scale=1)


def emm_gammalognorm(x, wgamma, a, s, loc, scale):
    return wgamma * gamma.pdf(x, a) + (1 - wgamma) * lognorm.pdf(x, s, loc, scale)


def bpl(x, norm, aup, adn, x0):
    return norm * (x ** (-adn)) / (1 + ((x / x0) ** (aup - adn)))


def pl(x, index):
    return x**index


def TimmerKonig_lightcurve_simulator(
    power_spectrum,
    npoints,
    spacing,
    nchunks=10,
    random_state="random-seed",
    power_spectrum_params=None,
    mean=0.0,
    std=1.0,
    noise=None,
    noise_type="gauss",
):
    if not callable(power_spectrum):
        raise ValueError(
            "The power spectrum has to be provided as a callable function."
        )

    if not isinstance(npoints * nchunks, int):
        raise TypeError("npoints and nchunks must be integers")

    random_state = get_random_state(random_state)

    npoints_ext = npoints * nchunks

    frequencies = np.fft.fftfreq(npoints_ext, spacing.value)

    real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))

    if power_spectrum_params:
        periodogram = power_spectrum(real_frequencies, **power_spectrum_params)
    else:
        periodogram = power_spectrum(real_frequencies)

    real_part = random_state.normal(0, 1, len(periodogram) - 1)
    imaginary_part = random_state.normal(0, 1, len(periodogram) - 1)

    if npoints_ext % 2 == 0:
        idx0 = -2
        random_factor = random_state.normal(0, 1)
    else:
        idx0 = -1
        random_factor = random_state.normal(0, 1) + 1j * random_state.normal(0, 1)

    fourier_coeffs = np.concatenate(
        [
            np.sqrt(0.5 * periodogram[:-1]) * (real_part + 1j * imaginary_part),
            np.sqrt(0.5 * periodogram[-1:]) * random_factor,
        ]
    )
    fourier_coeffs = np.concatenate(
        [fourier_coeffs, np.conjugate(fourier_coeffs[idx0::-1])]
    )

    fourier_coeffs = np.insert(fourier_coeffs, 0, 0)
    time_series = np.fft.ifft(fourier_coeffs).real

    ndiv = npoints_ext // (2 * nchunks)
    setstart = npoints_ext // 2 - ndiv
    setend = npoints_ext // 2 + ndiv
    if npoints % 2 != 0:
        setend = setend + 1
    time_series = time_series[setstart:setend]

    time_series = (time_series - time_series.mean()) / time_series.std()

    if noise:
        if noise_type == "gauss":
            noise_series = np.random.normal(loc=0, scale=noise, size=npoints)
        elif noise_type == "counts":
            noise_series = np.random.poisson(lam=noise, size=npoints)
        else:
            raise ValueError("Accepted values for 'noise_type' are 'gauss' or 'counts'")
        time_series += noise_series

    time_series = time_series * std + mean

    time_axis = np.linspace(0, npoints * spacing.value, npoints) * spacing.unit

    return time_series, time_axis


def Emmanoulopoulos_lightcurve_simulator(
    pdf,
    psd,
    npoints,
    spacing,
    pdf_params=None,
    psd_params=None,
    random_state="random-seed",
    imax=1000,
    nchunks=10,
    mean=0.0,
    std=1.0,
    noise=None,
    noise_type="gauss",
):
    lc_norm, taxis = TimmerKonig_lightcurve_simulator(
        psd,
        npoints,
        spacing,
        nchunks=nchunks,
        power_spectrum_params=psd_params,
        random_state=random_state,
    )

    random_state = get_random_state(random_state)

    fft_norm = np.fft.rfft(lc_norm)

    a_norm = np.abs(fft_norm) / npoints

    xx = np.linspace(0, 10, 10000)
    lc_sim = np.interp(
        random_state.rand(npoints),
        np.cumsum(pdf(xx, **pdf_params)) / np.sum(pdf(xx, **pdf_params)),
        xx,
    )

    nconv = True
    i = 0
    while nconv and i < imax:
        i += 1
        fft_sim = np.fft.rfft(lc_sim)
        phi_sim = np.angle(fft_sim)
        fft_adj = a_norm * np.exp(1j * phi_sim)
        lc_adj = np.fft.irfft(fft_adj, npoints)
        if np.array_equal(np.argsort(lc_sim), np.argsort(lc_adj)):
            nconv = False
        lc_adj[np.argsort(lc_adj)] = lc_sim[np.argsort(lc_sim)]
        lc_sim = lc_adj

    lc_sim = (lc_sim - lc_sim.mean()) / lc_sim.std()

    if noise:
        if noise_type == "gauss":
            noise_series = np.random.normal(loc=0, scale=noise, size=npoints)
        elif noise_type == "poisson":
            noise_series = np.random.poisson(lam=noise, size=npoints)
        else:
            raise ValueError(
                "Accepted values for 'noise_type' are 'gauss' or 'poisson'"
            )
        lc_sim += noise_series

    lc_sim = lc_sim * std + mean

    return lc_sim, taxis


def _generate_periodogram(args):
    """Helper function to generate a periodogram for a single realization."""
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


def _generate_histogram(args):
    """Helper function to generate a histogram for a single realization."""
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
        bins,
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
    hist, _ = np.histogram(tseries, bins=bins)
    return hist


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

    envelopes_psd = np.array(results)[..., 1 : npoints // 2 + 1]

    freqs = np.fft.fftfreq(npoints_ext, spacing_ext.value)[1 : npoints // 2 + 1]

    return envelopes_psd, freqs


def lightcurve_hist_envelope(
    pdf,
    psd,
    npoints,
    spacing,
    nsims=10000,
    pdf_params=None,
    psd_params=None,
    mean=0.0,
    std=1.0,
    noise=None,
    noise_type="gauss",
    bins=None,
):
    if bins is None:
        bins = int(10 ** np.floor(np.log10(npoints)))

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
            bins,
        )
        for _ in range(nsims)
    ]

    with Pool() as pool:
        results = pool.map(_generate_histogram, args)

    envelopes_hist = np.array(results)

    return envelopes_hist, bins


def _x2_fit_helper(
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
    hgram,
    bins,
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

    envelopes, bins = lightcurve_hist_envelope(
        pdf,
        psd,
        npoints,
        spacing,
        nsims=nsims,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
        noise=noise,
        noise_type=noise_type,
        bins=bins,
    )
    mean = np.nanmean(envelopes, axis=0)
    std = np.nanstd(envelopes, axis=0)
    std[std == 0.0] = np.sqrt(mean[std == 0]) / (nsims * (nsims - 1))
    std[std == 0.0] = 1
    obs = (hgram - mean) ** 2 / std**2
    sim = (envelopes - mean) ** 2 / std**2
    sumobs = np.nansum(obs)
    sumsim = np.nansum(sim, axis=-1)
    sign = len(np.where(sumobs >= sumsim)[0]) / nsims
    return sumobs * sign / len(obs)


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
    if not isinstance(nexp, int):
        raise TypeError(
            "The number of MC simulations for the error evaluation nexp must be an integer!"
        )
    kwargs.setdefault("method", "Powell")
    results = minimize(
        _x2_fit_helper,
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
    hgram,
    bins,
    psd,
    psd_params,
    pdf,
    pdf_initial,
    spacing,
    nsims=10000,
    mean=None,
    std=None,
    noise=None,
    noise_type="gauss",
    output_type="value",
    **kwargs,
):

    kwargs.setdefault("method", "Powell")
    results = minimize(
        _pdf_fit_helper,
        list(pdf_initial.values()),
        args=(
            hgram,
            bins,
            int(np.sum(hgram)),
            spacing,
            psd,
            psd_params,
            pdf,
            nsims,
            mean,
            std,
            noise,
            noise_type,
        ),
        **kwargs,
    )

    if output_type == "parameters":
        return results.x
    elif output_type == "value":
        return results.func
    elif output_type == "full":
        return results
    else:
        raise ValueError(
            f"Accepted values for {output_type} are 'parameters', 'value' or 'full'"
        )
