from setuptools import setup, find_packages

setup(
    name="barotropic",
    description="A framework for barotropic analysis and modelling of the atmosphere",
    version="3.0.0",
    author="Christopher Polster",
    url="https://github.com/chpolste/barotropic",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pyspharm"
    ],
    extras_require={
        "with-optional": [
            "matplotlib",
            "PyWavelets>=1.1.0.dev",
            "hn2016_falwa",
        ],
        "tests": [
            "pytest"
        ]
    }
)

