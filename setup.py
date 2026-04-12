from setuptools import find_packages, setup

# read the contents of README file
from os import path

# get __version__ from _version.py
ver_file = path.join('pyod', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='pyod',
    version=__version__,
    description='A Python library for anomaly detection across tabular, time series, graph, text, and image data. 60+ detectors, benchmark-backed ADEngine orchestration, and an agentic workflow for AI agents.',
    long_description=readme(),
    long_description_content_type='text/x-rst',
    author='Yue Zhao',
    author_email='yzhao062@gmail.com',
    url='https://github.com/yzhao062/pyod',
    keywords=[
        'anomaly detection',
        'outlier detection',
        'machine learning',
        'deep learning',
        'unsupervised learning',
        'time series anomaly detection',
        'graph anomaly detection',
        'nlp anomaly detection',
        'image anomaly detection',
        'multimodal',
        'agentic ai',
        'foundation models',
        'fraud detection',
        'novelty detection',
        'out-of-distribution detection',
        'outlier ensembles',
        'pytorch',
        'python',
    ],
    packages=find_packages(exclude=['test', 'test.*', 'pyod.test', 'pyod.test.*']),
    include_package_data=True,
    package_data={
        'pyod.utils.model_analysis_jsons': ['*.json'],
        'pyod.utils.knowledge': ['*.json'],
    },
    install_requires=requirements,
    extras_require={
        # Neural detectors (AutoEncoder, VAE, DeepSVDD, ALAD, ...)
        'torch': ['torch>=2.0'],
        # Acceleration
        'suod': ['suod'],
        # Supervised detector
        'xgboost': ['xgboost'],
        # Model combination utilities
        'combo': ['combo'],
        # Data-driven thresholding
        'pythresh': ['pythresh'],
        # EmbeddingOD paths
        'embedding': ['sentence-transformers>=5.0.0'],
        'openai': ['openai>=1.0'],
        'huggingface': ['transformers>=4.25.1', 'torch>=2.0', 'Pillow'],
        # Graph detectors (DOMINANT, CoLA, SCAN, ...)
        'graph': ['torch>=2.0', 'torch_geometric>=2.0'],
        # MCP server for agent integration
        'mcp': ['mcp>=1.0'],
        # Everything at once
        'all': [
            'torch>=2.0',
            'suod',
            'xgboost',
            'combo',
            'pythresh',
            'sentence-transformers>=5.0.0',
            'openai>=1.0',
            'transformers>=4.25.1',
            'torch_geometric>=2.0',
            'Pillow',
            'mcp>=1.0',
        ],
    },
    python_requires='>=3.9',
    license='BSD-2-Clause',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
