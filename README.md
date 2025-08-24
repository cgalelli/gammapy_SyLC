# gammapy_SyLC  

**gammapy_SyLC** is a Python package designed for time-domain analysis of high-energy astrophysical sources. It provides tools for simulating and fitting light curves, with a focus on Active Galactic Nuclei (AGN) variability. The package implements the **Timmer & König** and **Emmanoulopoulos** algorithms for light curve simulation, as well as **power spectral density (PSD) fitting** and **probability density function (PDF) fitting** functionalities.  

It is designed to be **compatible with [Gammapy](https://gammapy.org/)**, although it does not depend on it. The package enables users to perform statistical studies of variability, including PSD reconstruction, PDF fitting, and Monte Carlo-based parameter estimation.  

A short paper describing the package can be found in the paper.md file.

A longer paper, which also shows an application of the software to AGN lightcurves observed by Fermi-LAT, is available on **arXiv**: [https://arxiv.org/abs/2503.14156](https://arxiv.org/abs/2503.14156).  

---

## Features  

- **Light Curve Simulation**  
  - Generate synthetic light curves using the **Timmer & König** and **Emmanoulopoulos** algorithms  
  - Simulate light curves with a given **PSD** and **flux amplitude distribution**  
  - Introduce **noise** into simulations  

- **Variability Model Fitting**  
  - **Power Spectral Density (PSD) fitting** using Monte Carlo-based statistical envelopes  
  - **Flux amplitude distribution fitting** with lognormal, gamma, and alpha-stable models  
  - Tools for comparing **alternative statistical models**  

---

## Installation  

To install **gammapy_SyLC**, first clone the repository and install the package using `pip`.
```bash
git clone https://github.com/cgalelli/gammapy_SyLC.git
cd gammapy_SyLC
pip install .
```

### Required Dependencies  

Before using the package, ensure that the following dependencies are installed:  

- `numpy`  
- `scipy`  
- `astropy`
- `pytest`

Additionally, some external packages such as `matplotlib`, `pyLCR`, and `gammapy` are **not required** but can be useful for data visualization, retrieving Fermi-LAT light curves, and integrating with **gammapy** workflows.

---

## Quick Start

### Simulating a Light Curve

You can generate synthetic light curves using the Timmer & König or Emmanoulopoulos methods. The example below demonstrates how to generate a light curve with a power-law PSD and a lognormal flux distribution using the Emmanoulopoulos algorithm.

```python
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from gammapy_SyLC import *

# Define PSD and PDF models
psd_model = pl
pdf_model = lognormal

# Set parameters for the simulation
psd_params = {"index": -1.2}
pdf_params = {"s": 0.5}
obs_times = np.linspace(0, 7000, 1000) * u.d

# Generate synthetic light curve
tseries, times = Emmanoulopoulos_lightcurve_simulator(
    pdf_model, psd_model, obs_times,
    pdf_params=pdf_params, psd_params=psd_params,
    mean=1.0, std=0.5
)

# Plot the simulated light curve
plt.plot(times, tseries)
plt.xlabel("Time (days)")
plt.ylabel("Flux")
plt.title("Simulated Light Curve")
plt.show()
```

### Example: PSD Fitting for a Gamma-Ray Light Curve

Below is an example workflow using gammapy_SyLC to analyze a Fermi-LAT light curve retrieved with pyLCR.

```python
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.timeseries import LombScargle
from gammapy_SyLC import *
import pyLCR

# Retrieve a gamma-ray light curve from the Fermi-LAT Light Curve Repository
data = pyLCR.getLightCurve('4FGL J2202.7+4216', cadence='weekly', flux_type='photon', index_type='fixed', ts_min=4)
times = data.time
flux = data.flux
flux_err = data.flux_error

# Compute the periodogram
ls = LombScargle(times, flux, flux_err)
freq, power = ls.autopower(nyquist_factor=1, samples_per_peak=1, normalization="psd")

# Fit the PSD with a power-law model
psd_index = psd_fit(freq, power, times, pl, {"index": -1}, mean=flux.mean(), std=flux.std(), nsims=1000, nexp=-1)
print(f"Best-fit PSD index: {psd_index}")

# Generate PSD envelope for visualization
envelopes, freqs = lightcurve_psd_envelope(
    pl, times, psd_params={"index": psd_index},
    simulator="TK", nsims=1000, mean=flux.mean(), std=flux.std(),
)
qmin2, qmin1, qmax1, qmax2 = np.quantile(envelopes, [0.025, 0.16, 0.84, 0.975], axis=0)

# Plot periodogram and PSD envelope
plt.plot(freqs, np.median(envelopes, axis=0))
plt.fill_between(freqs, qmin2, qmax2, alpha=0.5, color='#1f77b4')
plt.fill_between(freqs, qmin1, qmax1, alpha=0.3, color='#1f77b4')
plt.plot(freqs, pgram[1:], linewidth=0.7, label="Observed")
plt.yscale("log")
plt.xlabel("Frequency (1/day)")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.show()
```

---


## Citation

If you use gammapy_SyLC in your research, please cite the associated paper:

 https://arxiv.org/abs/2503.14156

```yaml
 @article{Galelli2025,
    author = {C. Galelli},
    title = {gammapy_SyLC: A new tool to simulate and fit variability in high-energy light curves},
    journal = {arXiv},
    year = {2025},
    eprint = {2503.14156},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.IM}
}
```

## License

This project is licensed under the BSD 3-clause license. See the LICENSE file for details.

## Contact

For questions, please contact: Claudio Galelli – claudio.galelli@mi.infn.it
