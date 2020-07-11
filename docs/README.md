# Documentation

The HTML documentation is generated from the docstrings in the source code with [`pdoc`](https://github.com/mitmproxy/pdoc/releases/tag/0.3.2) and a modified template.


## Building

If you have all required packages installed, run

    ./build

in this folder to generate the html files.


## Updating the Online Documentation

The online documentation is served via GitHub pages on an orphaned `gh-pages` branch that contains only the html files.
To update the branch, run

    ./push

after generating the documentation locally.
The online documentation should only be updated when a new version is declared.

