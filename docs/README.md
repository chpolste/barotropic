# Documentation

The HTML documentation is generated from the docstrings in the source code with sphinx.


## Building

If you have all required packages installed, run

    make html

in this folder to generate the html files.


## Updating the Online Documentation

The online documentation is served via GitHub pages on an orphaned `gh-pages` branch that contains only the html files.
To update the branch, run

    ./push

after generating the documentation locally.
The online documentation should only be updated when a new version is declared.

