import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz
from regions import PointSkyRegion, CircleSkyRegion

from .simulators import ModifiedTimmerKonig_lightcurve_simulator, Emmanoulopoulos_lightcurve_simulator

def simulate_flux_points(
    psd_model,
    psd_params,
    obs_starts,
    livetimes,
    energy_axis,
    pointing_position,
    spectral_model,
    irfs,
    location,
    pdf_model=None,
    pdf_params=None,
    simu_radius=5*u.deg,
    oversample=10,
    simulator="EMM",
    random_state="random-seed",
    energy_axis_true=None,
    tref=Time("2000-01-01 00:00:00"),
):
    
    """
    Simulates flux points for a given spectral and temporal model.

    This function simulates a light curve from a given power spectral density (PSD)
    model and then uses this light curve to create a fake `gammapy` flux points.

    This function requires the `gammapy` package as optional dependency.
    A `RuntimeError` will be raised if it is not installed.

    Parameters:
    -----------
    psd : callable
        Target power spectral density function.       
    psd_params : dict
        Parameters for the PSD model.
    obs_starts : astropy.units.Quantity 
        The start times of the observations.
    livetimes : astropy.units.Quantity)
        List of livetimes for each observation.
    energy_axis : gammapy.maps.MapAxis 
        The energy axis for the flux points.
    pointing_position : astropy.coordinates.SkyCoord
        The pointing position of the observation.
    spectral_model : gammapy.modeling.models.SpectralModel
        The spectral model to be used for simulation.
    irfs : dict
        Instrument Response Functions.
    location : astropy.coordinates.EarthLocation)
        The location of the observatory.
    pdf_model : callable, optional 
        The probability density function (PDF) model. Required for the "EMM" simulator. Defaults to None.
    pdf_params : dict, optional)
        Parameters for the PDF model. Required for the "EMM" simulator. Defaults to None.
    simu_radius : astropy.units.Quantity, optional
        The radius of the simulated region. Defaults to 5 * u.deg.
    oversample : int, optional 
        The oversampling factor for the sampling times in the simulation. Defaults to 10.
    simulator : str, optional 
        The light curve simulator to use. Can be "EMM" or "MTK". Defaults to "EMM".
    random_state : int or 'random-seed', optional
        Random seed for reproducibility. Default is 'random-seed'.
    energy_axis_true : gammapy.maps.MapAxis, optional
        The true energy axis for the dataset.
    tref : astropy.time.Time, optional
        The reference time for the simulation. Defaults to "2000-01-01 00:00:00".

    Returns:
    --------
    simulated_lightcurve : gammapy.FluxPoints
        The simulated light curve.
    """

    try:
        # These are the required imports. If any of them fail, the ImportError will be caught.
        from gammapy.maps import MapAxis, RegionGeom, RegionNDMap
        from gammapy.estimators import LightCurveEstimator
        from gammapy.datasets import Datasets, SpectrumDataset
        from gammapy.makers import SpectrumDatasetMaker
        from gammapy.modeling.models import SkyModel, LightCurveTemplateTemporalModel
        from gammapy.data import Observation, FixedPointingInfo
    except ImportError as e:
        raise RuntimeError(
            "The `gammapy` and `regions` packages are required for this function. "
            "Please install them to use `simulate_flux_points`."
        ) from e

    livetimes.to(obs_starts.unit)

    model_obs_times = np.linspace(obs_starts[0], (obs_starts+livetimes)[-1], len(obs_starts)*oversample)

    time_axis = MapAxis.from_nodes(model_obs_times, name="time", interp="lin")

    if simulator == "EMM":
        if pdf_model is None or pdf_params is None:
            raise ValueError("`pdf_model` and `pdf_params` are required for EMM simulator.")
        model_sim, model_obs_times = Emmanoulopoulos_lightcurve_simulator(pdf_model, psd_model, model_obs_times, pdf_params=pdf_params, psd_params=psd_params, mean=1.2, std=0.6)

    else:
        model_sim, model_obs_times = ModifiedTimmerKonig_lightcurve_simulator(
            psd_model, model_obs_times, psd_params=psd_params, random_state=random_state, mean=1.2, std=0.6
            )
        
    geom = RegionGeom.create(
        CircleSkyRegion(pointing_position, simu_radius), axes=[energy_axis]
    )
    pointing = FixedPointingInfo(
        fixed_icrs=pointing_position.icrs,
    )

    m = RegionNDMap.create(
        region=PointSkyRegion(center=pointing_position),
        axes=[time_axis],
        unit="cm-2s-1TeV-1",
    )

    m.quantity = model_sim

    temporal_model = LightCurveTemplateTemporalModel(m, t_ref=tref)

    model_simu = SkyModel(
        spectral_model=spectral_model,
        temporal_model=temporal_model,
        name="model-simu",
    )

    tstart = obs_starts + tref

    altaz = pointing_position.transform_to(
        AltAz(obstime=tstart, location=location)
    )

    datasets = Datasets()

    if energy_axis_true is None:
        energy_axis_true = MapAxis.from_edges(np.logspace(-1.2, 2.0, 31), unit="TeV", name="energy_true", interp="log")

    empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="empty"
    )

    maker = SpectrumDatasetMaker(selection=["exposure", "background", "edisp"])

    for idx in range(len(tstart)):
        obs = Observation.create(
            pointing=pointing,
            livetime=livetimes[idx],
            tstart=tstart[idx],
            irfs=irfs,
            reference_time=tref,
            obs_id=idx,
            location=location,
        )
        empty_i = empty.copy(name=f"dataset-{idx}")
        dataset = maker.run(empty_i, obs)
        dataset.models = model_simu
        dataset.fake()
        datasets.append(dataset)

    model_fit = SkyModel(spectral_model=spectral_model, name="model-fit")
    datasets.models = model_fit

    lc_maker_1d = LightCurveEstimator(
        energy_edges=[energy_axis.edges[0].value, energy_axis.edges[-1].value]*energy_axis.unit,
        source="model-fit",
        selection_optional=["ul"],
    )

    simulated_lightcurve = lc_maker_1d.run(datasets)
    return simulated_lightcurve


def from_flux_points(flux_points):
    """
    Extracts numpy arrays for time, flux, and error from a `FluxPoints` object.
    """
    if not isinstance(flux_points, FluxPoints):
        raise TypeError("Input must be a gammapy.estimators.FluxPoints object.")

    times = flux_points.geom.axes["time"].time_edges
    
    if 'norm' in flux_points.colnames:
        flux = flux_points.norm.data.flatten()
        flux_err = flux_points.norm_err.data.flatten()
    elif 'dnde' in flux_points.colnames:
        flux = flux_points.dnde.data.flatten()
        flux_err = flux_points.dnde_err.data.flatten()
    else:
        raise ValueError("FluxPoints object must contain 'norm' or 'dnde' columns.")

    return times, flux, flux_err
