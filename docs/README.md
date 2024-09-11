# Documentation

The HTML documentation is generated from the docstrings in the source code with sphinx.

See the associated [Documentation action](../.github/workflows/docs.yml).


## Building

Make sure [Pandoc](https://pandoc.org/) and a Fortran compiler are available, e.g. by running

    sudo apt install gfortran pandoc

or an equivalent command for the package manager of your choice and operating system.

From the root of the repository, run:

    pip install .[extras,docs]
    pip install git+https://github.com/jswhit/pyspharm
    cd docs
    make html

The documentation website is generated in `docs/build/html`.

