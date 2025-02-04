import numbers
import numpy as np


def _random_state(init):
    """Get a `numpy.random.RandomState` instance.

    The purpose of this utility function is to have a flexible way
    to initialise a `~numpy.random.RandomState` instance,
    a.k.a. a random number generator (``rng``).

    Parameters
    ----------
    init : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Available options to initialise the RandomState object:

        * ``int`` -- new RandomState instance seeded with this integer
          (calls `~numpy.random.RandomState` with ``seed=init``)
        * ``'random-seed'`` -- new RandomState instance seeded in a random way
          (calls `~numpy.random.RandomState` with ``seed=None``)
        * ``'global-rng'``, return the RandomState singleton used by ``numpy.random``.
        * `~numpy.random.RandomState` -- do nothing, return the input.

    Returns
    -------
    random_state : `~numpy.random.RandomState`
        RandomState instance.
    """
    if isinstance(init, (numbers.Integral, np.integer)):
        return np.random.RandomState(init)
    elif init == "random-seed":
        return np.random.RandomState(None)
    elif init == "global-rng":
        return np.random.mtrand._rand
    elif isinstance(init, np.random.RandomState):
        return init
    else:
        raise ValueError(
            "{} cannot be used to seed a numpy.random.RandomState"
            " instance".format(init)
        )


def TimmerKonig_lightcurve_simulator(
        power_spectrum,
        npoints,
        spacing,
        nchunks=10,
        random_state="random-seed",
        psd_params=None,
        mean=0.0,
        std=1.0,
        noise=None,
        noise_type="gauss",
):
    """
    Simulate a light curve using the Timmer & Koenig method.

    Parameters:
    -----------
    power_spectrum : callable
        The target power spectrum as a function of frequency.
    npoints : int
        Number of points in the simulated light curve.
    spacing : astropy.units.Quantity
        Time spacing between successive points.
    nchunks : int, optional
        Oversampling factor. Default is 10.
    random_state : int or 'random-seed', optional
        Random seed for reproducibility. Default is 'random-seed'.
    psd_params : dict, optional
        Parameters for the power spectrum function. Default is None.
    mean : float, optional
        Desired mean of the light curve. Default is 0.0.
    std : float, optional
        Desired standard deviation of the light curve. Default is 1.0.
    noise : float or None, optional
        Noise amplitude to add to the light curve. Default is None.
    noise_type : {'gauss', 'counts'}, optional
        Type of noise to add. Default is 'gauss'.

    Returns:
    --------
    time_series : array_like
        The simulated light curve.
    time_axis : array_like
        Time values corresponding to the light curve.
    """
    if not callable(power_spectrum):
        raise ValueError(
            "The power spectrum has to be provided as a callable function."
        )

    if not isinstance(npoints * nchunks, int):
        raise TypeError("npoints and nchunks must be integers")

    random_state = _random_state(random_state)

    npoints_ext = npoints * nchunks

    frequencies = np.fft.fftfreq(npoints_ext, spacing.value)

    real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))

    if psd_params:
        periodogram = power_spectrum(real_frequencies, **psd_params)
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
    """
    Simulate a light curve using the Emmanoulopoulos method.

    Parameters:
    -----------
    pdf : callable
        Target probability density function for flux amplitudes.
    psd : callable
        Target power spectral density function.
    npoints : int
        Number of points in the simulated light curve.
    spacing : astropy.units.Quantity
        Time spacing between successive points.
    pdf_params : dict, optional
        Parameters for the PDF function. Default is None.
    psd_params : dict, optional
        Parameters for the PSD function. Default is None.
    random_state : int or 'random-seed', optional
        Random seed for reproducibility. Default is 'random-seed'.
    imax : int, optional
        Maximum number of iterations for convergence. Default is 1000.
    nchunks : int, optional
        Oversampling factor. Default is 10.
    mean : float, optional
        Desired mean of the light curve. Default is 0.0.
    std : float, optional
        Desired standard deviation of the light curve. Default is 1.0.
    noise : float or None, optional
        Noise amplitude to add to the light curve. Default is None.
    noise_type : {'gauss', 'poisson'}, optional
        Type of noise to add. Default is 'gauss'.

    Returns:
    --------
    lc_sim : array_like
        The simulated light curve.
    taxis : array_like
        Time values corresponding to the light curve.
    """
    lc_norm, taxis = TimmerKonig_lightcurve_simulator(
        psd,
        npoints,
        spacing,
        nchunks=nchunks,
        psd_params=psd_params,
        random_state=random_state,
    )

    random_state = _random_state(random_state)

    fft_norm = np.fft.rfft(lc_norm)

    a_norm = np.abs(fft_norm) / npoints

    xx = np.linspace(-10, 10, 10000)
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
