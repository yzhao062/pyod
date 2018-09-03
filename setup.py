from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

exec(open(path.join('pyod', 'version.py')).read())

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

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
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
