# Photoactivation Paper

This is a repository of analysis code for a paper in preparation. The
repository includes:

* Jupyter notebooks that perform all of the analysis, from the raw data to the
  published figure
* a Python file containing some functions shared by the notebooks
* an ``environment.yml`` file that specifies that exact versions of all
  Python packages used in the analysis
* other [YAML](https://en.wikipedia.org/wiki/YAML#Examples) files containing
  experiment metadata

It does not contain the large data files. These should be downloaded separately
from Google Drive or S3.

## Browse the notebooks

Review static (not runnable) copies of the notebooks [here](http://nbviewer.ipython.org/github/danielballan/photoactivation/tree/master).

## Installation

1. Download the data files and put them in the `data/` directory (no
   subdirectories).
2. Install [conda](https://www.continuum.io/downloads).
3. Open a terminal (command prompt) and use `cd` to change to the directory where these files are downloaded.
4. Install the required Python packages into a new "environment".

    ```
    conda-env create environment.yml
    ```

5. Activate the new environment. On Windows:

    ```
    activate pa
    ```

   On Mac/Linux:

    ```
    source activate pa
    ```
6. Start Jupyter notebook (formerly called IPython notebook).

    ```
    jupyter notebook
    ```

   This will open a browser tab where you can run and experiment with the
   notebooks.
