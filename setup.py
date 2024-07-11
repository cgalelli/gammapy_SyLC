from setuptools import find_packages, setup

setup(
    name="gammapy_SyLC",
    version="0.1.0",
    description="A tool for synthetic lightcurves for Gammapy",
    package_dir={"": "gammapy_SyLC"},
    packages=find_packages(where="gammapy_SyLC"),
    author="Claudio Galelli",
    author_email="<claudio.galelli@obspm.fr>",
    license="BSD 3-Clause",
    repository="https://github.com/cgalelli/gammapy_SyLC.git",
    readme="README.md",
    classifiers=["Topic :: Scientific/Engineering"],
    install_requires=["numpy", "gammapy", "scipy",],
    python_requires=">3.10",
)
