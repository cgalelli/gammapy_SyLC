import numpy as np

def _random_state(init):
    if isinstance(init, (np.integer)):
        return np.random.RandomState(init)
    elif init == "random-seed":
        return np.random.RandomState(None)
    elif init == "global-rng":
        return np.random.mtrand._rand
    elif isinstance(init, np.random.RandomState):
        return init
    else:
        raise ValueError(f"{init} cannot be used to seed a RandomState instance")

def TimmerKonig_lightcurve_simulator(
        power_spectrum,
        obs_times,
        nchunks=10,
        random_state="random-seed",
        psd_params=None,
        mean=0.0,
        std=1.0,
        nsims=1,
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
    nsims : int, optional
        Number of simulated light curves to generate. Default is 1.

    Returns:
    --------
    time_series : array_like
        The simulated light curve.
    time_axis : array_like
        Time values corresponding to the light curve.
    """
    if not callable(power_spectrum):
        raise ValueError("The power spectrum has to be provided as a callable function.")

    npoints = len(obs_times)
    spacing = np.diff(obs_times)[0]
    random_state = _random_state(random_state)
    npoints_ext = npoints * nchunks

    frequencies = np.fft.fftfreq(npoints_ext, spacing.value)
    real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))

    if psd_params:
        periodogram = power_spectrum(real_frequencies, **psd_params)
    else:
        periodogram = power_spectrum(real_frequencies)

    real_part = random_state.normal(0, 1, (nsims, len(periodogram) - 1))
    imaginary_part = random_state.normal(0, 1, (nsims, len(periodogram) - 1))

    if npoints_ext % 2 == 0:
        idx0 = -2
        random_factor = random_state.normal(0, 1, (nsims, 1))
    else:
        idx0 = -1
        random_factor = random_state.normal(0, 1, (nsims, 1)) + 1j * random_state.normal(0, 1, (nsims, 1))

    main_coeffs = np.sqrt(0.5 * periodogram[:-1]) * (real_part + 1j * imaginary_part)
    end_coeff = np.sqrt(0.5 * periodogram[-1:]) * random_factor
    fourier_coeffs = np.concatenate([main_coeffs, end_coeff], axis=1)
    
    neg_coeffs = np.conjugate(fourier_coeffs[:, idx0::-1])
    fourier_coeffs = np.concatenate([fourier_coeffs, neg_coeffs], axis=1)

    fourier_coeffs = np.insert(fourier_coeffs, 0, 0, axis=1)
    
    time_series = np.fft.ifft(fourier_coeffs, axis=1).real

    ndiv = npoints_ext // (2 * nchunks)
    setstart = npoints_ext // 2 - ndiv
    setend = npoints_ext // 2 + ndiv
    if npoints % 2 != 0:
        setend = setend + 1
        
    time_series = time_series[:, setstart:setend]

    ts_mean = time_series.mean(axis=1, keepdims=True)
    ts_std = time_series.std(axis=1, keepdims=True)
    time_series = (time_series - ts_mean) / ts_std
    time_series = time_series * std + mean

    if nsims == 1:
        time_series = time_series.flatten()

    return time_series, obs_times

def ModifiedTimmerKonig_lightcurve_simulator(
        power_spectrum,
        obs_times,
        psd_params=None,
        nchunks=10,
        random_state="random-seed",
        mean=0.0,
        std=1.0,
        nsims=1,
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
    nsims : int, optional
        Number of simulated light curves to generate. Default is 1.

    Returns
    -------
    time_series : array-like
        The simulated light curve values at the specified observation times.
    obs_times : array-like
        The input observation times.
    """
    random_state = _random_state(random_state)
    time_span = obs_times.max() - obs_times.min()
    n_freqs = (len(obs_times) // 2) * nchunks
    min_freq = 1.0 / time_span

    avg_spacing = np.mean(np.diff(obs_times))
    max_freq = 1.0 / (2.0 * avg_spacing)

    real_frequencies = np.linspace(min_freq, max_freq, n_freqs).value

    if psd_params:
        periodogram = power_spectrum(real_frequencies, **psd_params)
    else:
        periodogram = power_spectrum(real_frequencies)
    
    df = real_frequencies[1] - real_frequencies[0]
    rand_complex = random_state.normal(0, 1, (nsims, len(real_frequencies))) - 1j * random_state.normal(0, 1, (nsims, len(real_frequencies)))
    f_coeffs = np.sqrt(0.5 * periodogram * df) * rand_complex

    phase_matrix = np.exp(2j * np.pi * np.outer(real_frequencies, obs_times.value))
    
    time_series = (f_coeffs @ phase_matrix).real

    ts_mean = time_series.mean(axis=1, keepdims=True)
    ts_std = time_series.std(axis=1, keepdims=True)
    time_series = (time_series - ts_mean) / ts_std
    time_series = time_series * std + mean
    
    if nsims == 1:
        time_series = time_series.flatten()
        
    return time_series, obs_times

def Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        obs_times,
        pdf_params=None,
        psd_params=None,
        random_state="random-seed",
        base_sim="MTK",
        imax=1000,
        nchunks=10,
        mean=0.0,
        std=1.0,
        nsims=1,
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
    nsims : int, optional
        Number of simulated light curves to generate. Default is 1.

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
            psd, obs_times, nchunks=nchunks, psd_params=psd_params, random_state=random_state, nsims=nsims
        )
    elif base_sim == "TK":
        lc_norm, taxis = TimmerKonig_lightcurve_simulator(
            psd, obs_times, nchunks=nchunks, psd_params=psd_params, random_state=random_state, nsims=nsims
        )
    else:
        raise ValueError("Allowed values for base_sim are 'MTK' or 'TK'")

    random_state = _random_state(random_state)

    lc_norm = np.atleast_2d(lc_norm)

    fft_norm = np.fft.rfft(lc_norm, axis=1)
    a_norm = np.abs(fft_norm) / npoints

    x_min = max(1e-20, mean - 5*std) 
    x_max = mean + 50 * std
    xx = np.linspace(x_min, x_max, 10000)
    cdf = np.cumsum(pdf(xx, **pdf_params, mean=mean, std=std))
    cdf = cdf / cdf[-1]

    rand_vals = random_state.rand(nsims * npoints)
    lc_sim = np.interp(rand_vals, cdf, xx).reshape(nsims, npoints)

    row_idx = np.arange(nsims)[:, None]

    for i in range(imax):
        fft_sim = np.fft.rfft(lc_sim, axis=1)
        phi_sim = np.angle(fft_sim)
        fft_adj = a_norm * np.exp(1j * phi_sim)
        lc_adj = np.fft.irfft(fft_adj, npoints, axis=1)
        
        sort_sim = np.argsort(lc_sim, axis=1)
        sort_adj = np.argsort(lc_adj, axis=1)
        
        if np.array_equal(sort_sim, sort_adj):
            break
            
        lc_adj[row_idx, sort_adj] = lc_sim[row_idx, sort_sim]
        lc_sim = lc_adj

    if nsims == 1:
        lc_sim = lc_sim.flatten()

    return lc_sim, taxis