## Development

### Cloning

To work on `missingno` locally, you will need to clone it.

```git
git clone https://github.com/ResidentMario/missingno.git
```

You can then set up your own branch version of the code, and
 work on your changes for a pull request from there.

```bash
cd missingno
git checkout -B new-branch-name
```

### Environment

I strongly recommend creating a new virtual environment when working on `missingno` (e.g. not using the base system Python). You can do so with either [`conda`](https://conda.io/) or `virtualenv`. Once you have a virtual environment ready, I recommend running `pip install -e missingno .[tests]` from the root folder of the repository on your local machine.
This will create an [editable install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) of `missingno` suitable for tweaking and further development, and install additional test dependencies as well.

In addition to the `missingno` prerequisites, you will also need to have the `geopandas`, `geoplot`, and `shapely` optional dependencies installed (`geoplot` can only be installed via `conda`). To run the tests you will also need to have the `pytest` and `pytest-mpl` packages installed (this is done for you if you install with the `tests` flag set).

### Testing

Tests are split between visualization tests for core package methods and utility tests for helper functions.

The visualization tests are under `tests/viz_tests.py`, and rely on `pytest` and the [`pytest-mpl`](https://github.com/matplotlib/pytest-mpl) plugin. Before running the tests, generate the set of baseline images with `py.test --mpl-generate-path=baseline viz_tests.py`. You will want to inspect the images you generate in order to ascertain that they look correct. If you are satisfied that they are, you can rerun the test at anytime by calling `py.test --mpl viz_tests.py`. For more information on how this
works, refer to the [pytest-mpl README](https://github.com/matplotlib/pytest-mpl).

The utility tests are located under `tests/util_tests.py`. These are not visualization tests, and so can be run with `pytest util_tests.py`.

### Data

The `missingno` `README.md` uses a real-world data sample, one originally derived from the [NYPD Motor Vehicle Collisions Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95).
The data is zipped up and packaged online using [Quilt](https://quiltdata.com/package/ResidentMario/geoplot_data), a data-as-a-package management service. You can view the package [on the web](https://quiltdata.com/package/ResidentMario/missingno_data).

This data is only used in the `README.md`. It is not used for tests. Instructions for reading out the data are in the `README.md`; you can also do the following on your local machine:

```sh
$ pip install quilt
$ cd geoplot # root directory of this repo
$ quilt install # installs dependencies from quilt.yml
```

To update the data sample, you will need to push to this package, using [the instructions here](https://docs.quiltdata.com/make-a-package.html).

The data itself is a `csv` version of a `geojson` file (less geometry) packaged into the example data used by the `geoplot` package (another project of mine). The master copy is a separate repository on GitHub: [`missingno-data`](https://github.com/ResidentMario/missingno-data).

## Documentation

The Quickstart section of `README.md` is the principal documentation for this package. To edit the documentation I recommend editing that file directly on GitHub, which will handle generating a fork and pull request for you once your changes are made.