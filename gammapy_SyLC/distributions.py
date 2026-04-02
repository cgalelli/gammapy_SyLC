import numpy as np
from scipy.special import gammaln
from scipy.stats import levy_stable

def get_physical_bounds(mean, std, dist_type):
    """
    Returns the physical limit for the shape parameter to ensure loc >= 0.
    """
    cv = std / mean
    if dist_type == "lognormal":
        return np.sqrt(np.log(1 + cv**2))
    elif dist_type == "gamma":
        return (1.0 / cv)**2
    return

def lognormal(x, s, mean=1., std=1.):
    """
    Lognormal PDF mapped to a target mean and std.
    
    Parameters:
    -----------
    x : array_like
        Input values (fluxes).
    s : float
        Shape parameter (sigma of the underlying normal distribution).
    mean : float
        Target mean of the distribution.
    std : float
        Target standard deviation of the distribution.
    """
    s_min = get_physical_bounds(mean, std, "lognormal")
    if s < s_min:
        return np.zeros_like(x)
    
    term_s2 = np.exp(s**2)
    scale = std / np.sqrt(term_s2 * (term_s2 - 1))
    loc = mean - scale * np.exp(s**2 / 2)

    if loc < 0: return np.zeros_like(x)
    
    y = x - loc
    pdf = np.zeros_like(y, dtype=float)
    mask = y > 0
    
    log_y = np.log(y[mask])
    log_scale = np.log(scale)
    
    pdf[mask] = (1.0 / (y[mask] * s * np.sqrt(2 * np.pi))) * np.exp(
        -((log_y - log_scale)**2) / (2 * s**2)
    )
    
    return pdf


def gammaf(x, a, mean=1., std=1.):
    """
    Gamma PDF mapped to a target mean and std.
    
    Parameters:
    -----------
    x : array_like
        Input values (fluxes).
    a : float
        Shape parameter.
    mean : float
        Target mean of the distribution.
    std : float
        Target standard deviation of the distribution.
    """
    if a <= 0: return np.zeros_like(x)
    
    a_max = get_physical_bounds(mean, std, "gamma")
    if a > a_max:
        return np.zeros_like(x)
    theta = std / np.sqrt(a)
    loc = mean - a * theta

    if loc < 0: return np.zeros_like(x)
    
    y = x - loc
    pdf = np.zeros_like(y, dtype=float)
    mask = y > 0

    log_pdf = (a - 1) * np.log(y[mask]) - (y[mask] / theta) - gammaln(a) - a * np.log(theta)
    pdf[mask] = np.exp(log_pdf)
    
    return pdf


def maximal_alpha_stable(x, a, mean=1., std=1.):
    """
    Maximally skewed alpha-stable PDF.
    
    Note: For a < 2, these distributions have infinite variance. 
    Here, 'std' is used as a proxy for the scale parameter (gamma).
    'mean' is used as the location parameter (delta).
    """
    if a <= 0 or a > 2: return np.zeros_like(x)

    scale = std / np.sqrt(2) 
    loc = mean
    
    return levy_stable.pdf(x, a, 1, loc=loc, scale=scale)


def pl(x, index):
    """
    Simple power-law function, usually for PSD modeling.
    F(x) = x^index
    """
    return np.power(x, index)
