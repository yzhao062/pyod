from setuptools import find_packages, setup
from pyod import __version__

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyod',
    version=__version__,
    description='A Python Outlier Detection (Anomaly Detection) Toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yue Zhao',
    author_email='yuezhao@cs.toronto.edu',
    url='https://github.com/yzhao062/Pyod',
    download_url='https://github.com/yzhao062/Pyod/archive/master.zip',
    keywords=['outlier detection', 'anomaly detection', 'outlier ensembles',
              'data mining'],
    packages=find_packages(exclude=['examples,*test']),
    include_package_data=True,
    install_requires=[
        'numpy>=1.13',
        'scipy>=0.19.1',
        'scikit_learn>=0.19.1',
        'nose',
        'matplotlib',
    ],
    setup_requires=['setuptools>=38.6.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
