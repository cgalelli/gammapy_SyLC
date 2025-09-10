import numpy as np
import astropy.units as u
from astropy.time import Time
import pytest
from scipy.optimize import OptimizeResult

# Import all the necessary functions from SyLC
from gammapy_SyLC import (
    TimmerKonig_lightcurve_simulator,
    ModifiedTimmerKonig_lightcurve_simulator,
    Emmanoulopoulos_lightcurve_simulator,
    psd_fit,
    pdf_fit,
    lightcurve_psd_envelope,
    pl,
    lognormal,
    simulate_flux_points,
    from_flux_points,
)

# Check if gammapy is available, and skip tests if it's not
try:
    from gammapy.estimators import FluxPoints
    from gammapy.maps import MapAxis, RegionGeom
    GAMMAPY_AVAILABLE = True
except ImportError:
    GAMMAPY_AVAILABLE = False

print(GAMMAPY_AVAILABLE)

# --- Pytest Fixtures for reusable test data ---

@pytest.fixture(scope="module")
def common_params():
    """Provides a common set of parameters for simulations."""
    return {
        "psd_model": pl,
        "psd_params": {"index": -1.},
        "pdf_model": lognormal,
        "pdf_params": {"s": 0.5},
        "mean": 1.2,
        "std": 0.6,
    }

@pytest.fixture(scope="module")
def even_obs_times():
    """Provides a set of evenly sampled observation times."""
    return np.arange(100) * 7 * u.d

@pytest.fixture(scope="module")
def uneven_obs_times():
    """Provides a set of unevenly sampled observation times."""
    random_times = np.sort(np.random.uniform(0, 700, 100))
    return random_times * u.d

@pytest.fixture(scope="module")
def flux_points_object(even_obs_times):
    """Provides a gammapy FluxPoints object for testing compatibility."""
    if not GAMMAPY_AVAILABLE:
        return None
    
    time_axis = MapAxis.from_edges(even_obs_times, )
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    geom = RegionGeom.create(region=None, axes=[energy_axis, time_axis])
    
    fp = FluxPoints(geom=geom)
    fp['norm'] = np.random.randn(100, 1)
    fp['norm_err'] = np.random.rand(100, 1) * 0.1
    return fp


# --- Tests for Simulators ---

def test_timmerkonig_simulator_even(common_params, even_obs_times):
    """Test Timmer & Koenig simulator with evenly spaced data."""
    flux, times = TimmerKonig_lightcurve_simulator(
        common_params["psd_model"],
        obs_times=even_obs_times,
        psd_params=common_params["psd_params"],
    )
    assert len(flux) == len(even_obs_times)
    assert np.isfinite(flux).all()

def test_modified_timmerkonig_simulator_uneven(common_params, uneven_obs_times):
    """Test Modified Timmer & Koenig simulator with unevenly spaced data."""
    flux, times = ModifiedTimmerKonig_lightcurve_simulator(
        common_params["psd_model"],
        obs_times=uneven_obs_times,
        psd_params=common_params["psd_params"],
    )
    assert len(flux) == len(uneven_obs_times)
    assert np.isfinite(flux).all()

def test_emmanoulopoulos_simulator(common_params, even_obs_times):
    """Test Emmanoulopoulos light curve simulator."""
    flux, times = Emmanoulopoulos_lightcurve_simulator(
        common_params["pdf_model"],
        common_params["psd_model"],
        obs_times=even_obs_times,
        pdf_params=common_params["pdf_params"],
        psd_params=common_params["psd_params"],
    )
    assert len(flux) == len(even_obs_times)
    assert np.isfinite(flux).all()


# --- Tests for Fitting and Helper Functions ---

def test_psd_fit(common_params, uneven_obs_times):
    """Test PSD fitting on unevenly simulated data."""
    flux, _ = ModifiedTimmerKonig_lightcurve_simulator(
        common_params["psd_model"],
        uneven_obs_times,
        psd_params=common_params["psd_params"],
        mean=common_params["mean"],
        std=common_params["std"],
    )

    from astropy.timeseries import LombScargle
    ls = LombScargle(uneven_obs_times, flux, flux*0.3)
    freq, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")
    
    index_fit = psd_fit(
        frequencies=freq,
        power=power,
        obs_times=uneven_obs_times,
        psd=common_params["psd_model"],
        psd_initial=common_params["psd_params"],
        mean=common_params["mean"],
        std=common_params["std"],
        simulator="MTK",
        flux_error=flux*0.3,
        nsims=10,  # Reduced for speed in testing
        nexp=-1,    # Reduced for speed in testing
    )
    assert isinstance(index_fit, np.ndarray)

def test_pdf_fit(common_params, even_obs_times):
    """Test PDF fitting on simulated data."""
    flux, _ = Emmanoulopoulos_lightcurve_simulator(
        common_params["pdf_model"],
        common_params["psd_model"],
        obs_times=even_obs_times,
        pdf_params=common_params["pdf_params"],
        psd_params=common_params["psd_params"],
        mean=common_params["mean"],
        std=common_params["std"]
    )
    
    fit_result = pdf_fit(
        flux=flux,
        obs_times=even_obs_times,
        psd=common_params["psd_model"],
        psd_params=common_params["psd_params"],
        pdf=common_params["pdf_model"],
        pdf_initial=common_params["pdf_params"],
        nsims=100,
        output_type="full",
        bounds=[(0.1, None)]
    )
    assert isinstance(fit_result, OptimizeResult)
    assert fit_result.success

def test_lightcurve_psd_envelope(common_params, uneven_obs_times):
    """Test the PSD envelope generation."""
    flux_err = np.full(len(uneven_obs_times), 0.1)
    
    envelope, freqs = lightcurve_psd_envelope(
        psd=common_params["psd_model"],
        obs_times=uneven_obs_times,
        psd_params=common_params["psd_params"],
        mean=common_params["mean"],
        std=common_params["std"],
        flux_error=flux_err,
        simulator="MTK",
        nsims=10,
    )
    assert envelope.shape[0] == 10
    assert len(freqs) == envelope.shape[1]

# --- Tests for Gammapy Compatibility ---

@pytest.mark.skipif(not GAMMAPY_AVAILABLE, reason="gammapy not available")
class TestGammapyCompat:
    def test_from_flux_points(self, flux_points_object):
        """Test extracting data from a FluxPoints object."""
        times, flux, flux_err = from_flux_points(flux_points_object)
        assert isinstance(times, Time)
        assert isinstance(flux, np.ndarray)
        assert len(flux) == 100

    def test_simulate_flux_points(self, common_params):
        """A simple smoke test to ensure the gammapy simulation runs."""
        # This test requires a minimal set of gammapy objects
        from gammapy.irf import EDispKernel, EDispKernelMap
        from gammapy.modeling.models import PowerLawSpectralModel
        from gammapy.data import observatory_locations
        
        # A minimal set of IRFs and observation parameters
        energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
        energy_axis_true = MapAxis.from_energy_bounds("0.5 TeV", "20 TeV", nbin=4)
        edisp = EDispKernel.from_gauss(energy_axis_true=energy_axis_true, energy_axis=energy_axis, sigma=0.1, bias=0)
        edisp_map = EDispKernelMap(edisp)
        
        irfs = {'edisp': edisp_map, 'aeff': None, 'bkg': None} # Simplified for test
        
        obs_starts = np.arange(5) * u.h
        livetimes = np.full(5, 0.5) * u.h
        
        from astropy.coordinates import SkyCoord
        
        pointing = SkyCoord(ra=83.63, dec=22.01, unit='deg', frame="galactic")
        
        simulated_lc = simulate_flux_points(
            psd_model=common_params["psd_model"],
            psd_params=common_params["psd_params"],
            obs_starts=obs_starts,
            livetimes=livetimes,
            energy_axis=energy_axis,
            pointing_position=pointing,
            spectral_model=PowerLawSpectralModel(),
            irfs=irfs,
            location=observatory_locations["cta_south"], # Dummy location
            simulator="MTK",
            oversample=2 # Reduced for speed
        )
        assert isinstance(simulated_lc, FluxPoints)
        assert len(simulated_lc.geom.axes["time"]) == 4

if __name__ == "__main__":
    pytest.main()
