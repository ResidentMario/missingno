from setuptools import setup
setup(
  name = 'missingno',
  packages = ['missingno'], # this must be the same as the name above
  install_requires=['numpy', 'matplotlib', 'scipy', 'seaborn'],
  py_modules=['missingno'],
  version = '0.3.4',  # note to self: also update the one is the source!
  description = 'Missing data visualization module for Python.',
  author = 'Aleksey Bilogur',
  author_email = 'aleksey.bilogur@gmail.com',
  url = 'https://github.com/ResidentMario/missingno',
  download_url = 'https://github.com/ResidentMario/missingno/tarball/0.3.3',
  keywords = ['data', 'data visualization', 'data analysis', 'missing data', 'data science', 'pandas', 'python',
              'jupyter'],
  classifiers = [],
)