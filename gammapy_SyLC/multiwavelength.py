import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
from astropy.timeseries import LombScargle
from .simulators import ModifiedTimmerKonig_lightcurve_simulator, _random_state

def calculate_zdcf(t1, f1, e1, t2, f2, e2, lag_min, lag_max, lag_bin_width, min_pairs_per_bin=11):
    """
    Calculates the Z-transformed Discrete Correlation Function (ZDCF)
    for two unevenly sampled lightcurves.

    This implementation is based on the methodology described by
    T. Alexander (1997).

    Parameters:
    -----------
    t1, f1, e1 : array-like
        Time, flux, and flux error for the first lightcurve (e.g., reference band).
    t2, f2, e2 : array-like
        Time, flux, and flux error for the second lightcurve.
    lag_min : float
        The minimum time lag to consider.
    lag_max : float
        The maximum time lag to consider.
    lag_bin_width : float
        The width of the time lag bins.
    min_pairs_per_bin : int, optional
        The minimum number of pairs required to compute the correlation in a bin.
        Bins with fewer pairs will be discarded. Default is 11.

    Returns:
    --------
    dcf_results : structured array
        A structured numpy array containing the results for each valid bin, with
        the following fields:
        - 'lag': The central time lag of the bin.
        - 'dcf': The Discrete Correlation Function value (r).
        - 'dcf_pos_err': The upper 1-sigma error on the DCF.
        - 'dcf_neg_err': The lower 1-sigma error on the DCF.
        - 'n_pairs': The number of pairs in the bin.
    """

    mean1 = np.mean(f1)
    mean2 = np.mean(f2)
    std1 = np.std(f1, ddof=1)
    std2 = np.std(f2, ddof=1)
    mean_err1_sq = np.mean(e1**2)
    mean_err2_sq = np.mean(e2**2)

    denom_factor = np.sqrt(np.abs(std1**2 - mean_err1_sq) * np.abs(std2**2 - mean_err2_sq))
    # --- Step 2: Calculate all pairwise lags and UCCs ---
    dt = np.subtract.outer(t2, t1).flatten()
    
    # Calculate the Unbinned Correlation Coefficient (UCC) for all pairs
    ucc = np.outer((f1 - mean1), (f2 - mean2)).flatten() / denom_factor

    # --- Step 3: Define time lag bins ---
    lag_bins = np.arange(lag_min, lag_max + lag_bin_width, lag_bin_width)
    
    # --- Step 4: Bin the data and compute DCF and ZDCF ---
    bin_indices = np.digitize(dt, lag_bins)

    ucc_sum_per_bin = np.bincount(bin_indices, weights=ucc, minlength=len(lag_bins) + 1)
    n_pairs_per_bin = np.bincount(bin_indices, minlength=len(lag_bins) + 1)

    # The first and last elements correspond to values outside the lag_bins range, so we slice them off
    ucc_sum_per_bin = ucc_sum_per_bin[1:-1]
    n_pairs_per_bin = n_pairs_per_bin[1:-1]

    # Initialize result arrays with NaNs
    dcf = np.full(len(lag_bins) - 1, np.nan)
    dcf_pos_err = np.full(len(lag_bins) - 1, np.nan)
    dcf_neg_err = np.full(len(lag_bins) - 1, np.nan)

    # Filter for bins that meet the minimum pairs requirement
    valid_bins_mask = n_pairs_per_bin >= min_pairs_per_bin

    # Calculate the Discrete Correlation Function (DCF) for valid bins
    dcf[valid_bins_mask] = ucc_sum_per_bin[valid_bins_mask] / n_pairs_per_bin[valid_bins_mask]

    # --- Step 5: Apply Fisher's z-transformation for valid bins ---
    dcf_valid = dcf[valid_bins_mask]

    # Clamp DCF values to prevent log from returning infinity
    dcf_clamped = np.clip(dcf_valid, -0.99999, 0.99999)

    # Z-transformation
    z = 0.5 * np.log((1 + dcf_clamped) / (1 - dcf_clamped))
    z_err = 1.0 / np.sqrt(n_pairs_per_bin[valid_bins_mask] - 3)

    # Find 1-sigma confidence limits in z-space
    z_lower = z - z_err
    z_upper = z + z_err

    # Transform back to r-space
    dcf_lower = np.tanh(z_lower)
    dcf_upper = np.tanh(z_upper)

    # Calculate error bars
    dcf_neg_err[valid_bins_mask] = dcf_valid - dcf_lower
    dcf_pos_err[valid_bins_mask] = dcf_upper - dcf_valid
    
    return lag_bins, dcf, dcf_pos_err, dcf_neg_err, n_pairs_per_bin


def simulate_lightcurve_from_reference(
    t_ref, f_ref, e_ref,
    obs_times,
    mean_sim, std_sim,
    coherence, lag,
    nchunks=10, random_state="random-seed"
):
    """
    Simulates a lightcurve at one wavelength (t_sim) using an observed
    lightcurve from another wavelength (t_ref) as a reference.

    This method is ideal for using a well-sampled lightcurve to inform the
    simulation of a sparsely-sampled one, improving the constraints on its
    power spectrum.

    Parameters
    ----------
    t_ref, f_ref, e_ref : array-like
        Time, flux, and error of the observed reference lightcurve.
    t_sim : array-like
        The time points at which to simulate the new lightcurve.
    psd_sim, psd_params_sim, mean_sim, std_sim :
        PSD model, parameters, mean, and std for the new simulated lightcurve.
    coherence : callable
        The coherence between the two lightcurves as a function C(f) that returns coherence at a given frequency.
    lag : float or callable
        The time lag. Can be a single value (e.g., 10.0) or a function tau(f).
    nchunks : int, optional
        Oversampling factor for the frequency grid.
    random_state : int or 'random-seed', optional
        Seed for the random number generator.

    Returns
    -------
    lc_sim : array-like
        The simulated lightcurve at the t_sim time points.
    """
    random_state = _random_state(random_state)

    ls_ref = LombScargle(t_ref, f_ref, e_ref)
    
    time_span = obs_times.max() - obs_times.min()

    n_freqs = (len(obs_times) // 2) * nchunks
    min_freq = 1.0 / time_span

    avg_spacing = np.mean(np.diff(obs_times))
    max_freq = 1.0 / (2.0 * avg_spacing)

    freqs = np.linspace(min_freq, max_freq, n_freqs).value
    df = freqs[1] - freqs[0]

    # Fourier fingerprint of the reference LC: power and new random phases
    power_ref = ls_ref.power(freqs, normalization='psd')
    fourier_power_ref = power_ref * df
    phase_ref = 2 * np.pi * random_state.rand(n_freqs)
    f_coeffs_ref_complex = np.sqrt(fourier_power_ref) * np.exp(1j * phase_ref)

    coh_vals = coherence(freqs)
    lag_vals = np.full_like(freqs, lag)

    amp_sim = np.sqrt(fourier_power_ref)

    # Generate random coefficients for the incoherent part of the new LC
    # This part has the correct amplitude to match the target PSD after combination
    incoherent_amp = amp_sim * np.sqrt(1 - coh_vals**2)
    rand_phase = 2 * np.pi * random_state.rand(n_freqs)
    rand_coeffs_complex = incoherent_amp * np.exp(1j * rand_phase)

    # The coherent part is derived from the reference, scaled by coherence
    coherent_coeffs_complex = f_coeffs_ref_complex * coh_vals

    # Combine them and apply the time lag phase shift
    phase_lag = -2 * np.pi * freqs * lag_vals
    f_coeffs_sim_complex = (coherent_coeffs_complex + rand_coeffs_complex) * np.exp(1j * phase_lag)

    lc_sim = np.zeros(len(obs_times))
    for i in range(n_freqs):
        lc_sim += (f_coeffs_sim_complex[i] * np.exp(2j * np.pi * freqs[i] * obs_times.value)).real
    
    lc_sim = (lc_sim - np.mean(lc_sim)) / np.std(lc_sim) * std_sim + mean_sim
        
    return lc_sim, obs_times

def _mwl_psd_worker(args):
    """
    Worker function to simulate one lightcurve and compute its periodogram.
    """
    (
        t_ref, f_ref, e_ref, t_sim, f_sim, e_sim,
        psd_sim, psd_params_sim,
        coherence, lag, nchunks
    ) = args

    lc_sim_realization = simulate_lightcurve_from_reference(
        t_ref, f_ref, e_ref, t_sim,
        psd_sim, psd_params_sim, np.mean(f_sim), np.std(f_sim),
        coherence, lag, nchunks=nchunks
    )

    ls = LombScargle(t_sim, lc_sim_realization, e_sim)
    freqs, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")
    
    return freqs, power

def mwl_psd_envelope(
    t1, f1, e1,
    t2, f2, e2,
    psd2, psd_params2,
    coherence, lag,
    nsims=1000, nchunks=10
):
    """
    Generates a PSD envelope for a sparsely-sampled lightcurve (LC2) by using a
    well-sampled reference lightcurve (LC1) to inform the simulations.
    """
    arg_list = [(
        t1, f1, e1, t2, f2, e2,
        psd2, psd_params2,
        coherence, lag, nchunks
    ) for _ in range(nsims)]

    with Pool() as pool:
        results = pool.map(_mwl_psd_worker, arg_list)

    all_freqs, all_pgs = zip(*results)
    
    return np.array(all_pgs), all_freqs[0]