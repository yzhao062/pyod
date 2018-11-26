from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

exec(open(path.join('pyod', 'version.py')).read())
this_directory = path.abspath(path.dirname(__file__))

def readme():
    with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

# with open(path.join(this_directory, 'README_11262018.md'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='pyod',
    version=__version__,
    description='A Python Toolkit for Scalable Outlier Detection (Anomaly Detection)',
    long_description=readme(),  # commented out for now
    # long_description=long_description, # commented out for now
    # long_description_content_type='text/markdown', # commented out for now
    author='Yue Zhao',
    author_email='yuezhao@cs.toronto.edu',
    url='https://github.com/yzhao062/pyod',
    download_url='https://github.com/yzhao062/pyod/archive/master.zip',
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
        'Programming Language :: Python :: 3.7',
    ],
)
