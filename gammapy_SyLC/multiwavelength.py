import numpy as np
from astropy.timeseries import LombScargleMultiband
from .simulators import ModifiedTimmerKonig_lightcurve_simulator, TimmerKonig_lightcurve_simulator, Emmanoulopoulos_lightcurve_simulator

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


def _generate_mwl_periodogram(args):
    (
    simulator,
    pdf,
    psd,
    obs_times,
    known_times,
    known_fluxes,
    bands,
    frequencies,
    pdf_params,
    psd_params,
    mean,
    std,
    ) = args

    if not np.allclose(np.diff(obs_times), np.diff(obs_times)[0], rtol=1e-5) and simulator != "MTK":
        raise ValueError("Using an unevenly sampled 'obs_times' with a simulator that does not support it. Use simulator='MTK' for uneven observation times.")

    if len(bands) != len(known_times)+len(obs_times) or len(known_times) != len(known_fluxes):
        raise ValueError("Lengths don't match")

    if simulator == "TK":
        tseries, taxis = TimmerKonig_lightcurve_simulator(
            psd,
            obs_times,
            psd_params=psd_params,
            mean=mean,
            std=std,
        )
    elif simulator == "MTK":
        tseries, taxis = ModifiedTimmerKonig_lightcurve_simulator(
            psd,
            obs_times,
            psd_params=psd_params,
            mean=mean,
            std=std,
        )
    elif simulator == "EMM":
        tseries, taxis = Emmanoulopoulos_lightcurve_simulator(
            pdf,
            psd,
            obs_times,
            pdf_params=pdf_params,
            psd_params=psd_params,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError("Invalid simulator. Use 'TK', 'MTK' or 'EMM'.")

    full_times = np.concatenate((known_times, taxis))
    full_fluxes = np.concatenate((known_fluxes, tseries))

    ls = LombScargleMultiband(full_times, full_fluxes, bands)
    if frequencies is not None:
        power = ls.power(frequencies, normalization="psd")
    else:
        frequencies, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")

    return frequencies, power