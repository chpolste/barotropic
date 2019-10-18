from setuptools import setup, find_packages

setup(
    name="barotropic",
    description="A barotropic model on the sphere",
    version="1.0.0.devel",
    author="Christopher Polster",
    url="https://github.com/chpolste/barotropic",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyspharm"
    ],
    extras_require={
        "with-optional": [
            "matplotlib",
            "scipy",
            "PyWavelets>=1.1.0.dev",
            "hn2016_falwa",
        ]
    }
)

