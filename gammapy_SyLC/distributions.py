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
