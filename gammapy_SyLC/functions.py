import numpy as np
import inspect
from gammapy.stats import TimmerKonig_lightcurve_simulator
from gammapy.utils.random import get_random_state
from scipy.stats import gamma, lognorm
from scipy.signal import periodogram


def emm_gammalognorm(x, wgamma, a, s, loc, scale):
    return wgamma * gamma.pdf(x, a) + (1 - wgamma) * lognorm.pdf(x, s, loc, scale)


def bpl(x, norm, aup, adn, x0):
    return norm * (x ** (-adn)) / (1 + ((x / x0) ** (aup - adn)))


def pl(x, index):
    return x ** index


def Emmanoulopoulos_lightcurve_simulator(pdf, psd, npoints, spacing, pdf_params=None, psd_params=None,
                                         random_state="random-seed", imax=1000, nchunks=10, mean=0.0, std=1.0,
                                         poisson=False):
    target_cps = 0.2
    lc_norm, taxis = TimmerKonig_lightcurve_simulator(psd, npoints, spacing, nchunks=nchunks,
                                                      power_spectrum_params=psd_params, random_state=random_state)

    random_state = get_random_state(random_state)

    fft_norm = np.fft.rfft(lc_norm)

    a_norm = np.abs(fft_norm) / npoints

    if "scale" in pdf_params:
        scale = pdf_params.get("scale")
    else:
        scale = 1

    xx = np.linspace(0, scale * 10, 1000)
    lc_sim = np.interp(random_state.rand(npoints),
                       np.cumsum(pdf(xx, **pdf_params)) / np.sum(pdf(xx, **pdf_params)), xx)
    lc_sim = (lc_sim - lc_sim.mean()) / lc_sim.std()

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
    lc_sim = lc_sim * std + mean

    if poisson:
        lc_sim = (
                random_state.poisson(
                    np.where(lc_sim >= 0, lc_sim, 0) * spacing.decompose().value * target_cps
                )
                / (spacing.decompose().value * target_cps)
        )

    return lc_sim, taxis


def lightcurve_psd_envelope(psd, npoints, spacing, pdf=None, nsims=10000, pdf_params=None, psd_params=None,
                            simulator="TK", mean=0., std=1., oversample=10, poisson=False):
    npoints_ext = npoints * oversample
    spacing_ext = spacing / oversample
    if simulator == "TK":
        tseries, taxis = TimmerKonig_lightcurve_simulator(psd, npoints_ext, spacing_ext,
                                                          power_spectrum_params=psd_params, mean=mean, std=std,
                                                          poisson=poisson)
    elif simulator == "EMM":
        tseries, taxis = Emmanoulopoulos_lightcurve_simulator(pdf, psd, npoints_ext, spacing_ext, pdf_params=pdf_params,
                                                              psd_params=psd_params, mean=mean, std=std,
                                                              poisson=poisson)
    freqs, pg = periodogram(tseries, 1 / spacing_ext.value)
    envelopes_psd = np.empty((nsims, npoints // 2))
    envelopes_psd[0] = pg[1:npoints // 2 + 1]

    for _ in range(1, nsims):
        if simulator == "TK":
            tseries, taxis = TimmerKonig_lightcurve_simulator(psd, npoints_ext, spacing_ext,
                                                              power_spectrum_params=psd_params, mean=mean, std=std,
                                                              poisson=poisson)
        else:
            tseries, taxis = Emmanoulopoulos_lightcurve_simulator(pdf, psd, npoints_ext, spacing_ext,
                                                                  pdf_params=pdf_params, psd_params=psd_params,
                                                                  mean=mean, std=std, poisson=poisson)

        freqs, pg = periodogram(tseries, 1 / spacing_ext.value)
        envelopes_psd[_] = pg[1:npoints // 2 + 1]

    return envelopes_psd, freqs[1:npoints // 2 + 1]


def x2_fit(psd_params_list, pgram, npoints, spacing, psd, pdf=None, pdf_params=None, simulator="TK", nsims=10000,
           mean=None, std=None, poisson=False):
    psd_params_keys = list(inspect.signature(psd).parameters.keys())

    if len(psd_params_keys[1:]) != len(psd_params_list): raise ValueError(
        "parameter values do not correspond to the request from the psd function")

    psd_params = dict(zip(psd_params_keys[1:], psd_params_list))

    envelopes, freqs = lightcurve_psd_envelope(psd, npoints, spacing, pdf=pdf, pdf_params=pdf_params,
                                               psd_params=psd_params, simulator=simulator, nsims=nsims, mean=mean,
                                               std=std, poisson=poisson)

    if len(envelopes[0]) != len(pgram): raise ValueError("required length is different than data length!")

    obs = (pgram - envelopes.mean(axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sim = (envelopes - envelopes.mean(axis=0)) ** 2 / envelopes.std(axis=0) ** 2
    sumobs = np.sum(obs)
    sumsim = np.sum(sim, axis=-1)
    sign = len(np.where(sumobs >= sumsim)[0]) / nsims

    return sign
