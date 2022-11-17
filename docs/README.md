# Documentation

The HTML documentation is generated from the docstrings in the source code with sphinx.


## Building

If you have all required packages installed, run

    make html

in this folder to generate the html files.


## Updating the Online Documentation

The online documentation is served via GitHub pages on an orphaned `gh-pages` branch that contains only the html files.
It should only be updated when a new version is declared.

First, build the documentation from an up-to-date branch.
Delete and re-create the `gh-pages` branch:

    git branch -D gh-pages
    git switch --orphan gh-pages

Create a redirection from the `docs` folder to the build directory:

    touch .nojekyll
    echo "<meta http-equiv=\"Refresh\" content=\"0; url='build/html/'\" />" > index.html

Add all files necessary to serve the documentation:

    git add -f .nojekyll
    git add -f index.html
    git add -f build/html

Create a commit and push the documentation to GitHub:

    git commit --author "docs <docs@build.html>" -m "Add documentation page"
    git push -f origin gh-pages

