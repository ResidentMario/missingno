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

I strongly recommend creating a new virtual environment when working on `missingno` (e.g. not using the base system 
Python). You can do so with either [`conda`](https://conda.io/) or `virtualenv`. Once you have a virtual environment 
ready, I recommend running `pip install -e missingno .` from the root folder of the repository on your local machine.
This will create an [editable install](https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs) of 
`missingno` suitable for tweaking and further development.

### Testing

`missingno` is a data visualization package, and test suites for data visualization in Python are still rather 
finicky. An explicit test suite (likely using [`pytest-mpl`](https://github.com/matplotlib/pytest-mpl))is still a TODO.

## Documentation

The Quickstart section of `README.md` is the principal documentation for this package. To edit the documentation I 
recommend editing that file directly on GitHub, which will handle generating a fork and pull request for you once 
your changes are made.