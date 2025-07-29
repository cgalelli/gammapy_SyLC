import numpy as np
import astropy.units as u
import pytest
from scipy.optimize import OptimizeResult
from astropy.timeseries import LombScargle
from gammapy_SyLC import (
    TimmerKonig_lightcurve_simulator,
    Emmanoulopoulos_lightcurve_simulator,
    psd_fit,
    pdf_fit,
    compare_models,
    pl,
    lognormal,
    gammaf,
)


def test_timmerkonig_simulator():
    """Test Timmer & König light curve simulator."""
    psd = pl
    psd_params = {"index": -1.0}
    npoints = 50
    spacing = 7. * u.d

    flux, _ = TimmerKonig_lightcurve_simulator(
        psd,
        npoints,
        spacing,
        psd_params=psd_params,
    )

    assert len(flux) == npoints, "Incorrect number of points in simulated light curve"
    assert np.isfinite(flux).all(), "NaN or infinite values encountered in simulated light curve"


def test_emmanoulopoulos_simulator():
    """Test Emmanoulopoulos light curve simulator."""
    psd = pl
    psd_params = {"index": -1.4}
    pdf = lognormal
    pdf_params = {"s": 0.8}
    npoints = 10000
    spacing = 2. * u.s
    mean = 7.e-7
    std = 5.e-7

    flux, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
    )

    assert len(flux) == npoints, "Incorrect number of points in simulated light curve"
    assert np.isfinite(flux).all(), "NaN or infinite values encountered in simulated light curve"


def test_psd_fit():
    """Test PSD fitting on simulated data."""
    psd = pl
    psd_params = {"index": -1.0}
    npoints = 100
    spacing = 7. * u.d
    mean = 7.e-7
    std = 5.e-7

    flux, times = TimmerKonig_lightcurve_simulator(
        psd,
        npoints,
        spacing,
        psd_params=psd_params,
        mean=mean,
        std=std,
    )

    ls = LombScargle(times, flux)
    freq, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")
    index_fit, index_err = psd_fit(
        freq,
        power,
        psd,
        psd_params,
        spacing,
        mean=mean,
        std=std,
        nsims=100,
        nexp=10,
    )

    assert isinstance(index_fit, np.ndarray), "Unexpected output type from psd_fit"
    assert isinstance(index_err, np.ndarray)


def test_pdf_fit():
    """Test PDF fitting on simulated data."""
    psd = pl
    psd_params = {"index": -1.0}
    pdf = lognormal
    pdf_params = {"s": 0.5}
    npoints = 100
    spacing = 7. * u.d
    mean = 7.e-7
    std = 5.e-7

    flux, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
    )

    fit_result = pdf_fit(
        flux,
        psd,
        psd_params,
        pdf,
        pdf_params,
        spacing,
        nsims=50,
        output_type="full",
        bounds = [(0.1,None)],
    )

    assert isinstance(fit_result, (np.ndarray, OptimizeResult)), "Unexpected output type from pdf_fit"
    assert fit_result.success, "Optimization did not converge"


def test_model():
    """Test hypothesis testing against normal distribution."""
    psd = pl
    psd_params = {"index": -1.5}
    pdf = lognormal
    pdf_params = {"s": 0.5}
    npoints = 100
    spacing = 7. * u.d
    mean = 7.e-7
    std = 5.e-7

    flux, _ = Emmanoulopoulos_lightcurve_simulator(
        pdf,
        psd,
        npoints,
        spacing,
        pdf_params=pdf_params,
        psd_params=psd_params,
        mean=mean,
        std=std,
    )

    fit_stats, significance = compare_models(
        flux,
        pdf,
        pdf_params,
        psd,
        psd_params,
        gammaf,
        {"a": 0.9}
,        spacing,
        nsims=50,
        ntests=10,
        bounds = [(0.1,None)],
    )

    assert isinstance(significance, float), "Returned significance value is not a float"
    assert significance <= 1.0, "Significance value out of expected range"


if __name__ == "__main__":
    pytest.main()
