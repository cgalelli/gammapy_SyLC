from scipy.stats import gamma, lognorm, levy_stable


def lognormal(x, s):
    """
    Compute the probability density function (PDF) of the lognormal distribution.

    Parameters:
    -----------
    x : array_like
        Input values where the PDF is evaluated.
    s : float
        Shape parameter of the lognormal distribution.

    Returns:
    --------
    pdf_values : array_like
        The computed PDF values for the input `x`.
    """
    return lognorm.pdf(x, s, loc=0, scale=1)


def gammaf(x, a):
    """
    Compute the PDF of the gamma distribution.

    Parameters:
    -----------
    x : array_like
        Input values where the PDF is evaluated.
    a : float
        Shape parameter of the gamma distribution.

    Returns:
    --------
    pdf_values : array_like
        The computed PDF values for the input `x`.
    """
    if a <= 0: raise ValueError('Shape parameter must be strictly >0')
    return gamma.pdf(x, a, loc=0, scale=1)


def maximal_alpha_stable(x, a):
    """
    Compute the PDF of a maximally skewed alpha-stable distribution.

    Parameters:
    -----------
    x : array_like
        Input values where the PDF is evaluated.
    a : float
        Stability parameter (0 < a <= 2).

    Returns:
    --------
    pdf_values : array_like
        The computed PDF values for the input `x`.
    """
    if a <= 0: raise ValueError('Stability parameter must be strictly >0')

    return levy_stable.pdf(x, a, 1, loc=0, scale=1)


def emm_gammalognorm(x, wgamma, a, s, loc, scale):
    """
    Compute a weighted sum of gamma and lognormal PDFs.

    Parameters:
    -----------
    x : array_like
        Input values where the combined PDF is evaluated.
    wgamma : float
        Weight of the gamma component (0 <= wgamma <= 1).
    a : float
        Shape parameter for the gamma component.
    s : float
        Shape parameter for the lognormal component.
    loc : float
        Location parameter for the lognormal component.
    scale : float
        Scale parameter for the lognormal component.

    Returns:
    --------
    pdf_values : array_like
        The combined PDF values for the input `x`.
    """
    return wgamma * gamma.pdf(x, a) + (1 - wgamma) * lognorm.pdf(x, s, loc, scale)


def bpl(x, norm, aup, adn, x0):
    """
    Compute a broken power-law function.

    Parameters:
    -----------
    x : array_like
        Input values where the power-law is evaluated.
    norm : float
        Normalization constant.
    aup : float
        Power-law index for values above the break.
    adn : float
        Power-law index for values below the break.
    x0 : float
        Break position.

    Returns:
    --------
    values : array_like
        The computed broken power-law values for the input `x`.
    """
    return norm * (x ** (-adn)) / (1 + ((x / x0) ** (aup - adn)))


def pl(x, index):
    """
    Compute a simple power-law function.

    Parameters:
    -----------
    x : array_like
        Input values where the power-law is evaluated.
    index : float
        Power-law index.

    Returns:
    --------
    values : array_like
        The computed power-law values for the input `x`.
    """
    return x ** index
