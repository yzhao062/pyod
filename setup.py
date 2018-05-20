from distutils.core import setup

with open('Requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pyod',
    packages=['pyod'],
    version='0.1.0',
    description='A Python Toolbox for Outlier Detection (Anomaly Detection)',
    author='Yue Zhao',
    author_email='yuezhao@cs.toronto.edu',
    url='https://github.com/yzhao062/Pyod',
    download_url='https://github.com/yzhao062/Pyod/archive/master.zip',
    keywords=['outlier detection', 'anomaly detection', 'outlier ensembles'],
    install_requires=required,
    classifiers=[],
)
