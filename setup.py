from setuptools import setup
setup(
    name='missingno',
    license='MIT License',
    packages=['missingno'],
    install_requires=['numpy', 'matplotlib', 'scipy', 'seaborn'],
    extras_require={'tests': ['pytest', 'pytest-mpl']},
    py_modules=['missingno'],
    version='0.5.1',  # note to self: also update the one is the source!
    description='Missing data visualization module for Python.',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/missingno',
    download_url='https://github.com/ResidentMario/missingno/tarball/0.5.1',
    keywords=['data', 'data visualization', 'data analysis', 'missing data', 'data science', 'pandas', 'python',
              'jupyter'],
    classifiers=[]
)
