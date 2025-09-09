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
        obs_times,
        nchunks=10,
        random_state="random-seed",
        psd_params=None,
        mean=0.0,
        std=1.0,
):
    """
    Simulate a light curve using the Timmer & Koenig method.

    Parameters:
    -----------
    power_spectrum : callable
        The target power spectrum as a function of frequency.
    obs_times : astropy.units.Quantity
        Observation times. Needs to be evenly spaced; for unevenly spaced observation times, use ModifiedTimmerKonig_lightcurve_simulator().
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

    npoints = len(obs_times)
    spacing = np.diff(obs_times)[0]

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

    time_series = time_series * std + mean

    return time_series, obs_times

def ModifiedTimmerKonig_lightcurve_simulator(
        power_spectrum,
        obs_times,
        psd_params=None,
        nchunks=10,
        random_state="random-seed",
        mean=0.0,
        std=1.0,
):
    """
    Simulates a light curve at specific uneven time points using direct summation.

    Parameters
    ----------
    power_spectrum : callable
        A function that takes frequency (in 1/days) as input and returns the power
        spectral density (PSD) at that frequency.
    obs_times : array-like
        An array of observation times (in days) at which the light curve should be simulated.
    psd_params : dict, optional
        A dictionary of parameters to pass to the power_spectrum function.
    nchunks : int, optional
        The oversampling factor for the frequency grid. Default is 10.
    random_state : int or 'random-seed', optional
        Random seed for reproducibility. Default is 'random-seed'.
    mean : float, optional
        The desired mean of the simulated light curve. Default is 0.0.
    std : float, optional
        The desired standard deviation of the simulated light curve. Default is 1.0.

    Returns
    -------
    time_series : array-like
        The simulated light curve values at the specified observation times.
    obs_times : array-like
        The input observation times.
    """
    random_state = _random_state(random_state)

    # 1. Define a frequency grid
    # The number of frequencies is increased by the oversampling factor
    time_span = obs_times.max() - obs_times.min()

    n_freqs = (len(obs_times) // 2) * nchunks
    min_freq = 1.0 / time_span

    # Define max_freq based on the mean sampling, the "effective Nyquist"
    avg_spacing = np.mean(np.diff(obs_times))
    max_freq = 1.0 / (2.0 * avg_spacing)

    # Create the frequency array
    real_frequencies = np.linspace(min_freq, max_freq, n_freqs).value

    # 2. Compute the PSD values at these frequencies
    if psd_params:
        periodogram = power_spectrum(real_frequencies, **psd_params)
    else:
        periodogram = power_spectrum(real_frequencies)
    
    # 3. Generate random Fourier coefficients
    f_coeffs = np.sqrt(0.5 * periodogram* (real_frequencies[1] - real_frequencies[0]))*(random_state.normal(0, 1, len(real_frequencies))-1j*random_state.normal(0, 1, len(real_frequencies)))

    # 4. Direct summation to compute the time series at the desired observation times
    time_series = np.zeros(len(obs_times))
    for i in range(len(real_frequencies)):
        time_series += (f_coeffs[i] * np.exp(2j * np.pi * real_frequencies[i] * obs_times.value)).real


    # 5. Normalize the time series to have the desired mean and std
    time_series = (time_series - time_series.mean()) / time_series.std()
    time_series = time_series * std + mean
    
    return time_series, obs_times


def Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        obs_times,
        pdf_params=None,
        psd_params=None,
        random_state="random-seed",
        base_sim = "MTK",
        imax=1000,
        nchunks=10,
        mean=0.0,
        std=1.0,
):
    """
    Simulate a light curve using the Emmanoulopoulos method.

    Parameters:
    -----------
    pdf : callable
        Target probability density function for flux amplitudes.
    psd : callable
        Target power spectral density function.
    obs_times : astropy.units.Quantity
        Observation times. Needs to be evenly spaced.
    pdf_params : dict, optional
        Parameters for the PDF function. Default is None.
    psd_params : dict, optional
        Parameters for the PSD function. Default is None.
    random_state : int or 'random-seed', optional
        Random seed for reproducibility. Default is 'random-seed'.
    base_sim : "MTK" or "TK". Default is "MTK".
        Underlying simulator for base time series.
    imax : int, optional
        Maximum number of iterations for convergence. Default is 1000.
    nchunks : int, optional
        Oversampling factor. Default is 10.
    mean : float, optional
        Desired mean of the light curve. Default is 0.0.
    std : float, optional
        Desired standard deviation of the light curve. Default is 1.0.

    Returns:
    --------
    lc_sim : array_like
        The simulated light curve.
    taxis : array_like
        Time values corresponding to the light curve.
    """

    npoints = len(obs_times)

    if base_sim == "MTK":
        lc_norm, taxis = ModifiedTimmerKonig_lightcurve_simulator(
            psd,
            obs_times,
            nchunks=nchunks,
            psd_params=psd_params,
            random_state=random_state,
        )
    elif base_sim == "TK":
        lc_norm, taxis = TimmerKonig_lightcurve_simulator(
            psd,
            obs_times,
            nchunks=nchunks,
            psd_params=psd_params,
            random_state=random_state,
        )
    else:
        raise ValueError("Allowed values for base_sim are 'MTK' or 'TK'")

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

    lc_sim = lc_sim * std + mean

    return lc_sim, taxis
