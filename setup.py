from setuptools import find_packages, setup

setup(
    name="gammapy_SyLC",
    version="0.1.0",
    description="A tool for synthetic lightcurves for Gammapy",
    package_dir={"": "."},
    packages=find_packages(include=["gammapy_SyLC"]),
    author="Claudio Galelli",
    author_email="<claudio.galelli@obspm.fr>",
    license="BSD 3-Clause",
    repository="https://github.com/cgalelli/gammapy_SyLC.git",
    readme="README.md",
    classifiers=["Topic :: Scientific/Engineering"],
    install_requires=["numpy", "scipy", "astropy", "pytest"],
    python_requires=">3.10",
)
